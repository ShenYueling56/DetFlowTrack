import torch
import torch.nn as nn
import torch.nn.functional as F

from .sceneflow_head_template import SceneflowHeadTemplate
from .pwc_point_util import knn_point, index_points_group, PointConv, Conv1d
from pcdet.utils import box_utils
from pcdet.ops.pointnet2.pointnet2_3DSSD import pointnet2_utils

import torchsnooper

class CostVolumeLayer(nn.Module):
    def __init__(self, nsample, in_channel, mlp, use_det_feature=False, use_bn=False, use_leaky=True):
        super(CostVolumeLayer, self).__init__()
        self.nsample = nsample
        self.use_bn = use_bn
        self.use_det_feature = use_det_feature
        self.mlp_convs = nn.ModuleList()

        if use_bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if use_bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(negative_slope=0.1, inplace=True)

    # @torchsnooper.snoop()
    def forward(self, xyz1, xyz2, points1, points2, det_feature1=None, det_feature2=None):
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)
        # 在第二帧xyz2下找xyz1中每个点的k个最近邻,得到xyz2下的序列号
        knn_idx = knn_point(self.nsample, xyz2, xyz1)  # B, N1, nsample
        # 根据索引号得到xyz2下最近邻的xyz坐标
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        # 得到xyz2下最近邻和xuz1中每个点的距离
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)
        # xyz2下最近邻的feature
        grouped_points2 = index_points_group(points2, knn_idx)  # B, N1, nsample, D2
        # xyz1下各点的feature复制为和grouped_points2相同的维度
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        # 将xyz1的特征,xyz2下最近点的特征以及xyz2最近点和xyz1中心点距离拼接为新的特征new_points,并利用mlp进行特征提取
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim=-1)  # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1)  # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.use_bn:
                bn = self.mlp_bns[i]
                new_points = self.relu(bn(conv(new_points)))
            else:
                new_points = self.relu(conv(new_points))

        # weighted sum
        # 将xyz2最近点和xyz1中心点之间的距离作为特征聚合的权重
        if self.use_det_feature:
            # 得到最近邻的det_feature
            neighbor_det_fet = index_points_group(det_feature2, knn_idx)
            # det_feature的距离
            direction_det = neighbor_det_fet - det_feature1.view(B, N1, 1, C)
            direction = torch.cat([direction_xyz, direction_det], dim=-1)
        else:
            direction = direction_xyz
        dist = torch.norm(direction, dim=3).clamp(min=1e-10)  # B N1 nsample
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)  # B N1 1
        weights = (1.0 / dist) / norm  # B N1 nsample

        costVolume = torch.sum(weights.unsqueeze(-1).permute(0, 3, 2, 1) * new_points, dim=2)  # B C N
        return costVolume

class UpsamplingLayer(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        # import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1)  # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1)  # B S 3
        sparse_flow = sparse_flow.permute(0, 2, 1)  # B S 3
        knn_idx = knn_point(3, sparse_xyz, xyz)
        grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        dist = torch.norm(grouped_xyz_norm, dim=3).clamp(min=1e-10)
        norm = torch.sum(1.0 / dist, dim=2, keepdim=True)
        weight = (1.0 / dist) / norm

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim=2).permute(0, 2, 1)
        return dense_flow

class WarpingLayer(nn.Module):
    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        knn_idx = knn_point(3, xyz1_to_2, xyz2)
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm

        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2


class SceneFlowPredictor(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch=3, channels=[128, 128], mlp=[128, 64], neighbors=9, clamp=[-200, 200],
                 use_leaky=True):
        super(SceneFlowPredictor, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=True, use_leaky=True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out

        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    # @torchsnooper.snoop()
    def forward(self, xyz, feats, cost_volume, flow=None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim=1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim=1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])


class PWCPointHead(SceneflowHeadTemplate):
    """
    PointPWC
    input_channels:各层backbone提取的特征维度
    use_det_feature:是否将检测结果作为cost的注意力
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg)

        nsample = model_cfg.COST.NSAMPLE
        self.use_det_feature = model_cfg.USE_DET_FEATURE
        use_bn = model_cfg.USE_BN
        use_leaky = model_cfg.USE_LEAKY

        self.UpsampleLayer = UpsamplingLayer()
        self.WarpingLayer = WarpingLayer()
        self.CostVolumeModules = nn.ModuleList()
        self.SceneFlowPredictorModules = nn.ModuleList()

        # 最上层: todo:思考最上层是否还需要计算场景流???,因为最终的跟踪只需要框内的点就可以了??
        cost_volume_layer = CostVolumeLayer(nsample=nsample, in_channel=1*2+3, mlp=model_cfg.COST.MLP[1], use_det_feature=self.use_det_feature, use_bn=use_bn, use_leaky=use_leaky)
        scene_flow_predictor = SceneFlowPredictor(feat_ch=1+ model_cfg.FLOW.MLP[0][-1],
                                                          cost_ch=model_cfg.COST.MLP[1][-1],
                                                          channels=model_cfg.FLOW.CHANNELS[1], mlp=model_cfg.FLOW.MLP[
                        1])  # feat是拼接上flow_feat后的feat(除了最底层),需记录flow_feat,
        self.CostVolumeModules.append(cost_volume_layer)
        self.SceneFlowPredictorModules.append(scene_flow_predictor)

        # 从feature最高层1层到最底层
        for i in range(len(input_channels)):  # todo:迭代次数需要斟酌
            if input_channels[i] == -1:
                # cost_volume_layer = None
                # scene_flow_predictor = None
                # self.CostVolumeModules.append(cost_volume_layer)
                # self.SceneFlowPredictorModules.append(scene_flow_predictor)
                continue
            cost_volume_layer = CostVolumeLayer(nsample=nsample, in_channel=input_channels[i]*2+3, mlp=model_cfg.COST.MLP[i+1], use_det_feature=self.use_det_feature, use_bn=use_bn, use_leaky=use_leaky)
            if i == len(input_channels)-1:
                scene_flow_predictor = SceneFlowPredictor(feat_ch=input_channels[i],
                                                          cost_ch=model_cfg.COST.MLP[i+1][-1],
                                                          flow_ch=0,
                                                          channels=model_cfg.FLOW.CHANNELS[i+1], mlp=model_cfg.FLOW.MLP[
                        i+1])  # feat是拼接上flow_feat后的feat(除了最底层),需记录flow_feat,
            else:
                scene_flow_predictor = SceneFlowPredictor(feat_ch=input_channels[i] + model_cfg.FLOW.MLP[i][-1],
                                                          cost_ch=model_cfg.COST.MLP[i+1][-1],
                                                          channels=model_cfg.FLOW.CHANNELS[i+1], mlp=model_cfg.FLOW.MLP[
                        i+1])  # feat是拼接上flow_feat后的feat(除了最底层)

            self.CostVolumeModules.append(cost_volume_layer)
            self.SceneFlowPredictorModules.append(scene_flow_predictor)

    # todo: 使用mask,只对框内的点计算场景流
    # 根据gtboxes得到各层各个点的场景流的真实值标签
    # @torchsnooper.snoop()
    def assign_targets(self, input_dict1, input_dict2):
        """
        Args:
            input_dict1:
            input_dict2:

        Returns:
            targets_dict:
        """


        gt_boxes1 = input_dict1['gt_boxes']
        gt_boxes2 = input_dict2['gt_boxes']
        batch_size = gt_boxes1.shape[0]

        # 最顶层求场景流真实值
        xyz1_l0 = input_dict1['multi_xyzs'][0].contiguous().view(-1, 4)
        xyz2_l0 = input_dict2['multi_xyzs'][0].contiguous().view(-1, 4)

        targets_dict_points = self.assign_stack_targets(
            points1 = xyz1_l0,
            gt_boxes1=gt_boxes1,
            points2 = xyz2_l0,
            gt_boxes2 = gt_boxes2,
        )

        # 利用fs_idxs对各层分别处理
        fs_idxs1 = input_dict1['fs_idxs']
        targets_dict_points['multi_points_sceneflow_labels'] = [targets_dict_points['point_sceneflow_labels']]
        for i in range(len(fs_idxs1)-1):
            fs_idx1 = fs_idxs1[i]
            cur_point_sceneflow_labels = pointnet2_utils.gather_operation(
                    targets_dict_points['multi_points_sceneflow_labels'][-1], fs_idx1)
            targets_dict_points['multi_points_sceneflow_labels'].append(cur_point_sceneflow_labels)


        # 对center点计算sceneflow label
        centers1 = input_dict1['centers'].detach()
        centers2 = input_dict2['centers'].detach()
        assert centers1.shape.__len__() == 2, 'centers.shape=%s' % str(centers1.shape)
        targets_dict_center = self.assign_stack_targets(
            points1=centers1,
            gt_boxes1=gt_boxes1,
            points2=centers2,
            gt_boxes2=gt_boxes2,
        )
        targets_dict_points['center_sceneflow_labels'] = targets_dict_center['point_sceneflow_labels']
        targets_dict_points['multi_points_sceneflow_labels'].append(targets_dict_points['center_sceneflow_labels'])
        target_dict = targets_dict_points
        return target_dict

    # 获得多层场景流loss
    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        sceneflow_loss, tb_dict = self.get_sceneflow_loss()

        return sceneflow_loss, tb_dict

    # 获得场景流损失
    # @torchsnooper.snoop()
    def get_sceneflow_loss(self, tb_dict=None):
        multi_layer_flow_preds = self.flow_forward_ret_dict['multi_layer_flow_preds']
        multi_layer_flow_labels = self.flow_forward_ret_dict['multi_points_flow_labels']
        total_loss = torch.zeros(1).cuda()

        weight = self.model_cfg.LOSS.MULTI_LAVEL_WEIGHT
        if tb_dict is None:
            tb_dict = {}
        for i in range(len(multi_layer_flow_preds)):
            flow_preds = multi_layer_flow_preds[i] # B 3 N
            flow_labels = multi_layer_flow_labels[i]# B 3 N
            flow_diff = flow_preds - flow_labels
            loss =  weight[i] * torch.norm(flow_diff, dim=2).sum(dim=1).mean() #一帧下各点的flow的误差和, 并对一个batch求均值
            total_loss += loss
            tb_dict.update({"level%s_flow_loss"%str(i): loss})
        tb_dict.update({"total_flow_loss": total_loss})
        return total_loss, tb_dict

    # @torchsnooper.snoop()
    def forward(self, batch_dict1, batch_dict2):
        """
        Args:
            batch_dict:
                frame_id
                gt_boxes
                use_lead_xyz
                image_shape
                batch_size
                ctr_offsets
                centers
                centers_origin
                centers_features
                ctr_batch_idx
                fs_idxs        # 各层相对于上一层的采样index: B, npoint
                multi_features # backbone输出的各层feature
                multi_xyzs     # backbone输出的各层xyz  BN,4(batch_idx, x, y, z)
                cls_scores
                center_cls_scores
                batch_cls_preds   # encode后的最底层cls结果
                batch_box_preds   # encode后的最底层box结果
                batch_index
                cls_preds_normalized   # False
                multi_layer_cls_preds # 各层cls_head输出, 从最高层到最低层
                multi_layer_box_preds # 各层box_head输出, 从最高层到最低层
        Returns:
            batch_dict:

        """
        flow_preds = []
        flow_pred = None
        flow_feat = None
        features1 = batch_dict1['multi_features']  # level_num, B*N, feature_channel_num
        features2 = batch_dict2['multi_features']  # level_num, B*N, feature_channel_num
        xyzs1 = batch_dict1['multi_xyzs']   # level_num, B*N, 4
        xyzs2 = batch_dict2['multi_xyzs']   # level_num, B*N, 4
        batch_size = batch_dict1['batch_size']
        if self.use_det_feature:
            det_box_feat1 = batch_dict1['multi_layer_box_preds']
            det_cls_feat1 = batch_dict1['multi_layer_cls_preds']
            det_box_feat2 = batch_dict2['multi_layer_box_preds']
            det_cls_feat2 = batch_dict2['multi_layer_cls_preds']
        for i in range(len(self.SceneFlowPredictorModules)):
            feat1 = features1[-1 - i].view(batch_size, -1, features1[-1-i].shape[-1]).permute(0, 2, 1)
            feat2 = features2[-1 - i].view(batch_size, -1, features2[-1-i].shape[-1]).permute(0, 2, 1)
            xyz1 = xyzs1[-1 - i][:, 1:4].view(batch_size, -1, 3).permute(0, 2, 1)
            xyz2 = xyzs2[-1 - i][:, 1:4].view(batch_size, -1, 3).permute(0, 2, 1)
            det_feat1 = None
            det_feat2 = None
            if self.use_det_feature:
                det_feat1 = torch.cat([det_box_feat1[-1-i], det_cls_feat1[-1-i]], dim=-1)
                det_feat2 = torch.cat([det_box_feat2[-1-i], det_cls_feat2[-1-i]], dim=-1)

            # upsample
            if flow_pred is not None:
                flow_up = self.UpsampleLayer(xyz1,
                                        xyzs1[-i][:, 1:4].view(batch_size, -1, 3).permute(0, 2, 1).contiguous(),
                                        flow_pred.reshape(batch_size, -1, flow_pred.shape[1]).permute(0, 2, 1).contiguous()) # B 3 N
            else:
                flow_up = flow_pred
            # warping
            flow_warped = self.WarpingLayer(xyz1, xyz2, flow_up)
            # cost
            cost_volume_layer = self.CostVolumeModules[-1 - i]
            if self.use_det_feature:
                cost = cost_volume_layer(xyz1, flow_warped, feat1, feat2, det_feature1=det_feat1, det_feature2=det_feat2)
            else:
                cost = cost_volume_layer(xyz1, flow_warped, feat1, feat2)
            # sceneflow
            if flow_feat is not None:
                flow_feat_up = self.UpsampleLayer(xyz1, xyz2, flow_feat)
                new_feat = torch.cat([feat1, flow_feat_up], dim= 1)
            else:
                new_feat = feat1
            scene_flow_predictor = self.SceneFlowPredictorModules[-1 - i]
            flow_feat, flow_pred = scene_flow_predictor(xyz1, new_feat, cost, flow_up)
            flow_preds.insert(0, flow_pred) # 从最高层到最底层的flow   flow: B 3 N

        ret_dict = {
            'multi_layer_flow_preds': flow_preds,
            'center_flow_preds': flow_preds[-1],
        }

        batch_dict = {
            'frame1': batch_dict1,
            'frame2': batch_dict2,
        }
        batch_dict['flow_preds'] = flow_preds

        if self.training:
            target_dict = self.assign_targets(batch_dict1, batch_dict2) #获得多层场景流真实值
            ret_dict['multi_points_flow_labels'] = target_dict['multi_points_sceneflow_labels']

        self.flow_forward_ret_dict = ret_dict

        return batch_dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sceneflow_head_template import SceneflowHeadTemplate
from .pwc_point_util import knn_point, index_points_group, PointConv, Conv1d
from pcdet.utils import box_utils
from pcdet.ops.pointnet2.pointnet2_3DSSD import pointnet2_utils

import mayavi.mlab as mlab
from pcdet.utils.visual_utils import visualize_utils as V

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
    def forward(self, xyz1, xyz2, points1, points2, det_feature1=None, det_feature2=None, box1=None, box2=None):
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
            det_feature1 = det_feature1.reshape(B, N1, -1)
            det_feature2 = det_feature2.reshape(B, N2, -1)
            # 得到最近邻的det_feature
            neighbor_det_fet = index_points_group(det_feature2, knn_idx)
            # det_feature的距离
            direction_det = neighbor_det_fet - det_feature1.view(B, N1, 1, -1) #B, N, 1, 10
            dist_xyz = torch.norm(direction_xyz, dim=3).clamp(min=1e-10)  # B N1 nsample
            norm_xyz = torch.sum(1.0 / dist_xyz, dim=2, keepdim=True)  # B N1 1
            weights_xyz = (1.0 / dist_xyz) / norm_xyz  # B N1 nsample
            dist_det = torch.norm(direction_det, dim=3).clamp(min=1e-10)  # B N1 nsample
            norm_det = torch.sum(1.0 / dist_det, dim=2, keepdim=True)  # B N1 1
            weights_det = (1.0 / dist_det) / norm_det  # B N1 nsample
            weights = (weights_xyz + weights_det) / 2
            # direction = torch.cat([direction_xyz, direction_det], dim=-1)
        else:
            direction = direction_xyz
            dist = torch.norm(direction, dim=3).clamp(min=1e-10)  # B N1 nsample
            norm = torch.sum(1.0 / dist, dim=2, keepdim=True)  # B N1 1
            weights = (1.0 / dist) / norm  # B N1 nsample

        # # 权重可视化
        # print(xyz1.shape)
        # print(neighbor_xyz.shape)
        # print(weights.shape)
        # corners1 = box_utils.boxes_to_corners_3d(box1)
        # print(corners1.shape)
        #
        # inbox_flag = box_utils.in_hull(xyz1[0].cpu(), corners1[0].cpu())
        # print(inbox_flag.shape)
        # for index, center in enumerate(xyz1[0]):
        #     if inbox_flag[index] == 0:
        #         continue
        #     neighbor = neighbor_xyz[0][index, :, :]
        #     neighbor_weight1 = weights_xyz[0][index, :]
        #     neighbor_weight2 = weights[0][index, :]
        #     print(neighbor_weight2)
        #     neighbor1 = torch.cat([neighbor, neighbor_weight1.reshape(-1, 1)], -1)
        #     neighbor2 = torch.cat([neighbor, neighbor_weight2.reshape(-1, 1)], -1)
        #     V.draw_three_scenes(center.reshape(1, 3), points2=xyz1[0], points3=neighbor1, boxes1=box1, boxes2=box2)
        #     V.draw_three_scenes(center.reshape(1,3), points2=xyz1[0], points3=neighbor2, boxes1=box1, boxes2=box2)
        #     mlab.show(stop=True)


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


class FlowPointHead(SceneflowHeadTemplate):
    """
    FlowPointHead: 单层场景流
    input_channels:各层backbone提取的特征维度
    """
    def __init__(self, model_cfg, input_channels):
        super().__init__(model_cfg=model_cfg)

        self.use_det_feature = model_cfg.USE_DET_FEATURE
        use_bn = model_cfg.USE_BN
        use_leaky = model_cfg.USE_LEAKY

        # center层进行场景流学习
        self.inter_cost_volume_layer = None
        self.intra_cost_volume_layer = None
        last_channel = input_channels[-1]
        for cost_layer_cfg in model_cfg.COST:
            if cost_layer_cfg.NAME == 'InterCostVolume':
                print("inter cost volume")
                self.cost_volume_layer = CostVolumeLayer(cost_layer_cfg.NSAMPLE, in_channel=last_channel * 2 + 3,
                                                         mlp=cost_layer_cfg.MLP, use_det_feature=self.use_det_feature,
                                                         use_bn=use_bn, use_leaky=use_leaky)
            elif cost_layer_cfg.NAME == 'IntraCostVolume':
                self.intra_cost_volume_layer = CostVolumeLayer(cost_layer_cfg.NSAMPLE, in_channel=last_channel * 2 + 3,
                                                               mlp=cost_layer_cfg.MLP,
                                                               use_det_feature=self.use_det_feature, use_bn=use_bn,
                                                               use_leaky=use_leaky)
            last_channel = cost_layer_cfg.MLP[-1]
        # self.cost_volume_layer = CostVolumeLayer(nsample=nsample, in_channel=input_channels[-1]*2+3, mlp=model_cfg.COST.MLP, use_det_feature=self.use_det_feature, use_bn=use_bn, use_leaky=use_leaky)

        self.scene_flow_predictor = SceneFlowPredictor(feat_ch=input_channels[-1],
                                                      cost_ch=last_channel,
                                                      flow_ch=0,
                                                      channels=model_cfg.FLOW.CHANNELS, mlp=model_cfg.FLOW.MLP)  # feat是拼接上flow_feat后的feat(除了最底层)

    # 根据gtboxes得到center层,各个点的场景流的真实值标签
    # @torchsnooper.snoop()
    def assign_targets(self, input_dict1, input_dict2):
        """
        Args:
            input_dict1:
            input_dict2:

        Returns:
            target_dict:
        """

        gt_boxes1 = input_dict1['gt_boxes']
        gt_boxes2 = input_dict2['gt_boxes']
        batch_size = gt_boxes1.shape[0]

        # 对center点计算sceneflow label
        centers1 = input_dict1['centers'].detach()
        centers2 = input_dict2['centers'].detach()
        N = centers2.shape[0]
        centers1 = centers1[0:N, :]
        # print("centers1 ", centers1.shape)
        # print("centers2 ", centers2.shape)
        assert centers1.shape.__len__() == 2, 'centers.shape=%s' % str(centers1.shape)
        target_dict = {}
        targets_dict_center = self.assign_stack_targets(
            points1=centers1,
            gt_boxes1=gt_boxes1,
            points2=centers2,
            gt_boxes2=gt_boxes2,
        )
        target_dict['center_sceneflow_labels'] = targets_dict_center['point_sceneflow_labels']
        target_dict['center_inbox_labels'] = targets_dict_center['point_inbox_labels']
        # # 可视化场景流
        # # 对各个batch分别操作
        # bs_idx1 = centers1[:, 0]
        # bs_idx2 = centers2[:, 0]
        # for k in range(batch_size):
        #     bs_mask1 = (bs_idx1 == k)
        #     bs_mask2 = (bs_idx2 == k)
        #     points_single1 = centers1[bs_mask1][:, 1:4]
        #     points_single2 = centers2[bs_mask2][:, 1:4]
        #     flow_pred = target_dict['center_sceneflow_labels'][k].permute(1, 0)
        #     inbox_label = target_dict['center_inbox_labels'][k] > 0
        #     points_single12 = points_single1 + flow_pred
        #     gt_boxes_single1 = gt_boxes1[k][:,:7]
        #     gt_boxes_single2 = gt_boxes2[k][:,:7]
        #
        #     # 验证场景流标签正确性
        #     print("白色第一帧点,红色第二帧点,绿色场景流作用后的点,蓝色第一帧框,绿色第二帧框")
        #     V.draw_three_scenes(points_single1, points2=points_single2, points3=points_single12, boxes1=gt_boxes_single1, boxes2=gt_boxes_single2)
        #     mlab.show(stop=True)
        # #     #　验证框内mask标签正确性
        #     V.draw_scenes(points_single1[inbox_label], gt_boxes=gt_boxes_single1)
        #     mlab.show(stop=True)
        return target_dict

    # 获得场景流loss
    def get_loss(self, ret_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        sceneflow_loss, tb_dict = self.get_sceneflow_loss(ret_dict, tb_dict)

        return sceneflow_loss, tb_dict

    # 获得单层场景流损失
    # @torchsnooper.snoop()

    def get_sceneflow_loss(self, ret_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        flow_preds = ret_dict['center_flow_pred'].permute(0, 2, 1) # B N 3
        flow_labels = ret_dict['center_flow_labels'].permute(0, 2, 1) # B N 3
        inbox_mask = ret_dict['center_inbox_labels'].float() # B N

        inbox_num = inbox_mask.sum()
        inbox_num = torch.clamp(inbox_num, min=1.0)
        # print("inbox_num %s " % str(inbox_num))

        if tb_dict is None:
            tb_dict = {}
        flow_diff = flow_preds - flow_labels
        total_loss = inbox_mask * torch.norm(flow_diff, dim=2)
        total_loss = total_loss.sum(dim=1)
        total_loss = total_loss.sum() #一帧下各点的flow的误差和, 和各个batch的误差和
        total_loss = total_loss/inbox_num
        tb_dict.update({"total_flow_loss": total_loss})
        return total_loss, tb_dict

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
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
        batch_dict1 = batch_dict['frame1']
        batch_dict2 = batch_dict['frame2']
        N = batch_dict2['centers'].shape[0]
        feat1 = batch_dict1['centers_flow_features']  # *N, feature_channel_num
        feat2 = batch_dict2['centers_flow_features']  # level_num, B*N, feature_channel_num
        xyz1 = batch_dict1['centers'][0:N, :]   # level_num, B*N, 4
        xyz2 = batch_dict2['centers']   # level_num, B*N, 4
        # print("xyz1 ", xyz1.shape)
        # print("xyz2 ", xyz2.shape)
        # print("feat1 ", feat1.shape)
        # print("feat2 ", feat2.shape)

        batch_size = batch_dict1['batch_size']

        det_feat1 = None
        det_feat2 = None
        if self.use_det_feature:
            det_box_feat1 = batch_dict1['batch_point_box_preds']
            det_cls_feat1 = batch_dict1['batch_point_cls_preds']
            det_box_feat2 = batch_dict2['batch_point_box_preds']
            det_cls_feat2 = batch_dict2['batch_point_cls_preds']
            det_feat1 = torch.cat([det_box_feat1, det_cls_feat1], dim=-1) # B*N, 10
            det_feat2 = torch.cat([det_box_feat2, det_cls_feat2], dim=-1) # B*N, 10
            # print("det_feat1 ", det_feat1.shape)
            # print("det_feat2 ", det_feat2.shape)

        feat1 = feat1.view(batch_size, -1, feat1[-1].shape[-1]).permute(0, 2, 1)
        feat2 = feat2.view(batch_size, -1, feat2[-1].shape[-1]).permute(0, 2, 1)
        xyz1 = xyz1[:, 1:4].view(batch_size, -1, 3).permute(0, 2, 1)
        xyz2 = xyz2[:, 1:4].view(batch_size, -1, 3).permute(0, 2, 1)

        # cost
        if self.use_det_feature:
            gt_boxes1 = batch_dict1['gt_boxes'][0][:,:7]
            gt_boxes2 = batch_dict2['gt_boxes'][0][:,:7]
            cost = self.cost_volume_layer(xyz1, xyz2, feat1, feat2, det_feature1=det_feat1, det_feature2=det_feat2, box1=gt_boxes1, box2=gt_boxes2)
        else:
            cost = self.cost_volume_layer(xyz1, xyz2, feat1, feat2)
        if self.intra_cost_volume_layer is not None:
            if self.use_det_feature:
                cost = self.intra_cost_volume_layer(xyz1, xyz1, cost, cost, det_feature1=det_feat1, det_feature2=det_feat1)
            else:
                cost = self.intra_cost_volume_layer(xyz1, xyz1, cost, cost)
        # sceneflow
        flow_feat, flow_pred = self.scene_flow_predictor(xyz1, feat1, cost)

        ret_dict = {
            'center_flow_pred': flow_pred,
        }
        # print("center_flow_pred ", flow_pred.shape)

        batch_dict = {
            'frame1': batch_dict1,
            'frame2': batch_dict2,
        }
        batch_dict['center_flow_pred'] = flow_pred # B, 3, N

        # if self.training:
        if 'gt_boxes' in batch_dict1:  # 在eval阶段也提供scenflow的真实值以对sceneflow结果进行eval
            # print("batch_dict1 key", batch_dict1.keys())
            target_dict = self.assign_targets(batch_dict1, batch_dict2) #获得多层场景流真实值
            ret_dict['center_flow_labels'] = target_dict['center_sceneflow_labels']
            ret_dict['center_inbox_labels'] = target_dict['center_inbox_labels']

        self.flow_forward_ret_dict = ret_dict
        return batch_dict, ret_dict

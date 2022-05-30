import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
import torchsnooper

class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        if 'FLOW_FC' in self.model_cfg:
            # flow_pre_channel = c_out
            flow_pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
            flow_fc_list = []
            for k in range(0, self.model_cfg.FLOW_FC.__len__()):
                flow_fc_list.extend([
                    nn.Conv1d(flow_pre_channel, self.model_cfg.FLOW_FC[k], kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.model_cfg.FLOW_FC[k]),
                    nn.ReLU()
                ])
                flow_pre_channel = self.model_cfg.FLOW_FC[k]

                if k != self.model_cfg.FLOW_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    flow_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

            self.flow_fc_layer = nn.Sequential(*flow_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    # @torchsnooper.snoop()
    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
g
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        batch_index = torch.range(0, batch_size-1, device=global_roi_grid_points.device).reshape(-1, 1).repeat(1, global_roi_grid_points.shape[1]).reshape(-1, 1)
        roi_grid_points = torch.cat((batch_index, global_roi_grid_points.contiguous().view(-1, 3)), 1) # (B*N*6*6*6, 4)


        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return roi_grid_points, pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_points_roi_preds(self, batch_cls_preds, batch_box_preds, flow_points, rois):
        batch_size = batch_cls_preds.shape[0]
        # print("batch_cls_preds\n", batch_cls_preds.shape)  # (B, roi_num, 1)
        # print("batch_box_preds\n", batch_box_preds.shape)  # (B, roi_num, 7)
        kp_num = int(flow_points.shape[0] / batch_size)
        batch_point_cls_preds = torch.zeros((batch_size, kp_num, 1), device=flow_points.device)  # (B, kp_num, 1)
        batch_point_box_preds = torch.zeros((batch_size, kp_num, 7), device=flow_points.device)  # (B, kp_num, 7)

        # print("targets_dict['rois']\n", rois.shape)
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(flow_points[:, 1:4].view(batch_size, -1, 3),
                                                                    rois)  # (B, kp_num)
        # print("box_idxs_of_pts\n", box_idxs_of_pts.shape)
        for i in range(batch_size):
            batch_cls_preds_single = batch_cls_preds[i, :, :]  # (roi_num, 1)
            batch_box_preds_single = batch_box_preds[i, :, :]  # (roi_num, 7)
            box_idxs_of_pts_single = box_idxs_of_pts[i, :]  #
            box_idxs_of_pts_single_valid = box_idxs_of_pts_single[box_idxs_of_pts_single >= 0].long()
            # print("box_idxs_of_pts_single_valid\n", box_idxs_of_pts_single_valid)
            batch_point_cls_preds_single = batch_point_cls_preds[i, :, :]
            batch_point_box_preds_single = batch_point_box_preds[i, :, :]
            batch_point_cls_preds_single[box_idxs_of_pts_single >= 0] = batch_cls_preds_single[
                box_idxs_of_pts_single_valid]
            batch_point_box_preds_single[box_idxs_of_pts_single >= 0] = batch_box_preds_single[
                box_idxs_of_pts_single_valid]
            # print("batch_point_cls_preds_single\n", batch_point_cls_preds_single.shape)
            # print("batch_point_box_preds_single\n", batch_point_box_preds_single.shape)
            batch_point_cls_preds[i, :, :] = batch_point_cls_preds_single
            batch_point_box_preds[i, :, :] = batch_point_box_preds_single
        # print("batch_point_cls_preds\n", batch_point_cls_preds.shape)
        # print("batch_point_box_preds\n", batch_point_box_preds.shape)
        return batch_point_cls_preds, batch_point_box_preds

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        # 得到RoIs
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_points, pooled_features = self.roi_grid_pool(batch_dict)  # (B, Nx6x6x6, 3), (BxN, 6x6x6, C)

        # 获得rois中心点坐标
        batch_size = batch_dict['batch_size']
        # roi_center_points = targets_dict['rois'][:, :, 0:3] #(B*N, 3)
        # batch_index = torch.range(0, batch_size - 1, device=roi_center_points.device).reshape(-1, 1).repeat(1, roi_center_points.shape[1]).reshape(-1, 1)
        # roi_points = torch.cat((batch_index, roi_center_points.contiguous().view(-1, 3)), 1)  # (B*N, 4)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)


        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))  # (B*N, C, 1)
        # # 计算roi grid points的场景流
        # flow_features = self.flow_fc_layer(pooled_features.view(batch_size_rcnn, pooled_features.shape[1], -1)).permute(0, 2, 1).contiguous()
        # flow_features = flow_features.view(-1, flow_features.shape[-1]) # (B*N*216, c)
        # # 计算rois中心点的场景流
        # flow_features = self.flow_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1)).permute(0, 2, 1).contiguous()
        # flow_features = flow_features.view(-1, flow_features.shape[-1])  # (B*N, c)
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            # rois refine之后的预测框
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
            # # 使用rois的中心点或者grid points计算场景流
            # batch_dict['centers'] = roi_points
            # batch_dict['centers_flow_features'] = flow_features
            # # 特征点对应的cls_preds 和 box_preds
            flow_points = batch_dict['point_coords']  # (B*kp_num, 4)
            rois = targets_dict['rois']  # (B, roi_num, 7)
            batch_point_cls_preds, batch_point_box_preds = self.get_points_roi_preds(batch_cls_preds, batch_box_preds,
                                                                                     flow_points, rois)
            batch_dict['batch_point_cls_preds'] = batch_point_cls_preds
            batch_dict['batch_point_box_preds'] = batch_point_box_preds

        else:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            # # 特征点对应的cls_preds 和 box_preds
            flow_points = batch_dict['point_coords']  # (B*kp_num, 4)
            rois = targets_dict['rois']  # (B, roi_num, 7)
            batch_point_cls_preds, batch_point_box_preds = self.get_points_roi_preds(batch_cls_preds, batch_box_preds, flow_points, rois)
            batch_dict['batch_point_cls_preds'] = batch_point_cls_preds
            batch_dict['batch_point_box_preds'] = batch_point_box_preds
            # # 使用rois的中心点或者grid points计算场景流
            # batch_dict['centers'] = roi_points
            # batch_dict['centers_flow_features'] = flow_features
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            # self.forward_ret_dict = targets_dict

        return batch_dict, targets_dict

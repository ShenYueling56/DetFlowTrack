import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_template import PointHeadTemplate
from ...ops.pointnet2.pointnet2_3DSSD import pointnet2_utils
import torchsnooper

class  PointHeadMultiScaleBox3DSSD(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        # 默认情况下不生成每个点的track标签
        self.get_track_label = False

        # box coder
        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        # head
        cls_output_channels = num_class
        box_output_channels = self.box_coder.code_size
        self.cls_head_modules = nn.ModuleList()
        self.box_head_modules = nn.ModuleList()
        ## layer0
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC[0],
            input_channels=1+cls_output_channels,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC[0],
            input_channels=1+box_output_channels,
            output_channels=self.box_coder.code_size
        )
        self.cls_head_modules.append(self.cls_center_layers)
        self.box_head_modules.append(self.box_center_layers)

        for i in range(len(input_channels)-1):
            if input_channels[i] == -1:
                continue
            self.cls_center_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.CLS_FC[i+1],
                input_channels=input_channels[i]+cls_output_channels,
                output_channels=num_class
            )
            self.box_center_layers = self.make_fc_layers(
                fc_cfg=self.model_cfg.REG_FC[i+1],
                input_channels=input_channels[i]+box_output_channels,
                output_channels=self.box_coder.code_size
            )
            self.cls_head_modules.append(self.cls_center_layers)
            self.box_head_modules.append(self.box_center_layers)

        # 最底层 head
        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC[-1],
            input_channels=input_channels[-1],
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC[-1],
            input_channels=input_channels[-1],
            output_channels=self.box_coder.code_size
        )
        self.cls_head_modules.append(self.cls_center_layers)
        self.box_head_modules.append(self.box_center_layers)

    # def assign_targets(self, input_dict):
    #     """
    #     Args:
    #         input_dict:
    #             point_features: (N1 + N2 + N3 + ..., C)
    #             batch_size:
    #             point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
    #             gt_boxes (optional): (B, M, 8)
    #     Returns:
    #         point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
    #         point_part_labels: (N1 + N2 + N3 + ..., 3)
    #     """
    #     # point_coords = input_dict['point_coords']
    #     gt_boxes = input_dict['gt_boxes']
    #     # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    #     # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)
    #
    #     batch_size = gt_boxes.shape[0]
    #     extend_gt_boxes = box_utils.enlarge_box3d(
    #         gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
    #     ).view(batch_size, -1, gt_boxes.shape[-1])
    #
    #     centers = input_dict['centers'].detach()
    #     assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
    #     assert centers.shape.__len__() in [2], 'points.shape=%s' % str(centers.shape)
    #     targets_dict_center = self.assign_stack_targets(
    #         points=centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
    #         set_ignore_flag=True, use_ball_constraint=False,
    #         ret_part_labels=False, ret_box_labels=True
    #     )
    #     targets_dict_center['center_gt_box_of_fg_points'] = targets_dict_center['gt_box_of_fg_points']
    #     targets_dict_center['center_cls_labels'] = targets_dict_center['point_cls_labels']
    #     targets_dict_center['center_box_labels'] = targets_dict_center['point_box_labels']
    #     # print("targets_dict_center['center_gt_box_of_fg_points']: \n"+ str(targets_dict_center['center_gt_box_of_fg_points']))
    #     # print("targets_dict_center['center_cls_labels']: \n"+str(targets_dict_center['center_cls_labels']))
    #     # print("targets_dict_center['center_box_labels']: \n" + str(targets_dict_center['center_box_labels']))
    #     targets_dict = targets_dict_center
    #
    #     return targets_dict

    # @torchsnooper.snoop()
    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        # point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        # assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        # assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        # 对box进行扩大,以获得ignore label
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        # 最上层点(16384)
        xyz_l0 = input_dict['multi_xyzs'][0].contiguous().view(-1, 4)
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert xyz_l0.shape.__len__() in [2], 'points.shape=%s' % str(xyz_l0.shape)
        # targets_dict_points = self.multi_scale_assign_stack_targets(
        # targets_dict_points = self.assign_stack_targets(
        #     points=xyz_l0, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
        #     set_ignore_flag=True, use_ball_constraint=False,
        #     ret_part_labels=False, ret_box_labels=True, ret_track_labels=self.get_track_label,
        # )
        targets_dict_points = self.multi_scale_assign_stack_targets(
            points=xyz_l0, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True, ret_track_labels=self.get_track_label
        )

        # 根据fs_idx获得除最底层外各层的标签
        fs_idxs = input_dict['fs_idxs']
        targets_dict_points['multi_point_cls_labels'] = [targets_dict_points['point_cls_labels']] # B*N
        targets_dict_points['multi_point_box_labels'] = [targets_dict_points['point_box_labels']] # B*N, 8
        targets_dict_points['multi_point_part_labels'] = [targets_dict_points['point_part_labels']] # None
        targets_dict_points['multi_gt_box_of_fg_points'] = [targets_dict_points['gt_box_of_fg_points']] #pos_num, 8 (gt_box7维+cls1维)
        # print("targets_dict_points['gt_box_of_fg_points'] shape: "+str(targets_dict_points['gt_box_of_fg_points'].shape))

        for i in range(len(fs_idxs)-1):
            fs_idx = fs_idxs[i]
            # [B C N] , [B, npoints] -> [B, C, npoints]
            # print("point_cls_labels type: " + str(targets_dict_points['multi_point_cls_labels'][-1].dtype))
            cur_point_cls_labels = pointnet2_utils.gather_operation(
                targets_dict_points['multi_point_cls_labels'][-1].reshape(batch_size, -1, 1).permute(0, 2, 1).float().contiguous(), fs_idx).contiguous().reshape(-1).long()
            cur_point_box_labels = pointnet2_utils.gather_operation(
                targets_dict_points['multi_point_box_labels'][-1].reshape(batch_size, -1, 8).permute(0, 2, 1).contiguous(), fs_idx).contiguous().permute(0, 2, 1).reshape(-1, 8)
            if targets_dict_points['multi_point_part_labels'][-1] is not None:
                cur_point_part_labels = pointnet2_utils.gather_operation(
                    targets_dict_points['multi_point_part_labels'][-1].reshape(batch_size, -1, 3).permute(0, 2, 1).contiguous(), fs_idx).contiguous().permute(0, 2, 1).reshape(-1, 3)
            cur_gt_box_of_fg_points = pointnet2_utils.gather_operation(
                targets_dict_points['multi_gt_box_of_fg_points'][-1].reshape(batch_size, -1, 8).permute(0, 2, 1).contiguous(),
                fs_idx).contiguous().permute(0, 2, 1).reshape(-1, 8)

            # print("cur_gt_box_of_fg_points shape: " + str(cur_gt_box_of_fg_points.shape))

            targets_dict_points['multi_point_cls_labels'].append(cur_point_cls_labels)
            targets_dict_points['multi_point_box_labels'].append(cur_point_box_labels)
            if targets_dict_points['multi_point_part_labels'][-1] is not None:
                targets_dict_points['multi_point_part_labels'].append(cur_point_part_labels)
            targets_dict_points['multi_gt_box_of_fg_points'].append(cur_gt_box_of_fg_points)

        # 对center点分配gt label
        centers = input_dict['centers'].detach()
        assert centers.shape.__len__() in [2], 'points.shape=%s' % str(centers.shape)
        targets_dict_center = self.assign_stack_targets(
            points=centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True, ret_track_labels=self.get_track_label,
        )

        targets_dict_points['center_gt_box_of_fg_points'] = targets_dict_center['gt_box_of_fg_points']
        targets_dict_points['center_cls_labels'] = targets_dict_center['point_cls_labels']
        targets_dict_points['center_box_labels'] = targets_dict_center['point_box_labels']
        targets_dict_points['multi_gt_box_of_fg_points'].append(targets_dict_points['center_gt_box_of_fg_points'])
        targets_dict_points['multi_point_cls_labels'].append(targets_dict_points['center_cls_labels'])
        targets_dict_points['multi_point_box_labels'].append(targets_dict_points['center_box_labels'])

        targets_dict = targets_dict_points

        return targets_dict

    def upsample(self, unknown_points, known_points, known_feats):
        """
        args:
        unknown_points: B N 3
        known_points: B M 3
        known_feats: B C M
        return:
        interpolated_feats: B C N
        """
        dist, idx = pointnet2_utils.three_nn(unknown_points, known_points)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = pointnet2_utils.three_interpolate(known_feats, idx, weight)
        return interpolated_feats

    def get_loss(self, ret_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        # point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        center_loss_reg, tb_dict_3 = self.get_center_reg_layer_loss(ret_dict)
        center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss(ret_dict)
        center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss(ret_dict)
        corner_loss, tb_dict_6 = self.get_corner_layer_loss(ret_dict)

        point_loss = center_loss_reg + center_loss_cls + center_loss_box + corner_loss
        # point_loss = center_loss_reg + center_loss_cls + center_loss_box

        # tb_dict.update(tb_dict_1)
        # tb_dict.update(tb_dict_2)
        tb_dict.update(tb_dict_3)
        tb_dict.update(tb_dict_4)
        tb_dict.update(tb_dict_5)
        tb_dict.update(tb_dict_6)
        return point_loss, tb_dict

    # 框中心点loss 监督ctr_offset,不需要多层
    def get_center_reg_layer_loss(self, ret_dict, tb_dict=None):
        # 得到框内点mask
        pos_mask = ret_dict['center_cls_labels'] > 0
        # 框中心点label
        center_box_labels = ret_dict['center_gt_box_of_fg_points'][:, 0:3].clone().detach()
        centers_origin = ret_dict['centers_origin']
        ctr_offsets = ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        center_loss_box = F.smooth_l1_loss(
            centers_pred, center_box_labels
        )

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_reg': center_loss_box.item()})
        return center_loss_box, tb_dict

    # def get_center_cls_layer_loss(self, tb_dict=None):
    #     point_cls_labels = self.forward_ret_dict['center_cls_labels'].view(-1)
    #     point_cls_preds = self.forward_ret_dict['center_cls_preds'].view(-1, self.num_class)
    #
    #     positives = (point_cls_labels > 0)
    #     negative_cls_weights = (point_cls_labels == 0) * 1.0
    #     cls_weights = (negative_cls_weights + 1.0 * positives).float()
    #     pos_normalizer = positives.sum(dim=0).float()
    #     cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    #
    #     one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
    #     one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
    #     one_hot_targets = one_hot_targets[..., 1:]
    #
    #     if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
    #         centerness_mask = self.generate_center_ness_mask()
    #         one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])
    #         cls_loss_src = loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(point_cls_preds, one_hot_targets)
    #         cls_loss_src = cls_loss_src * cls_weights.unsqueeze(-1)
    #     else:
    #         cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
    #     point_loss_cls = cls_loss_src.sum()
    #
    #     loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
    #     point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
    #     if tb_dict is None:
    #         tb_dict = {}
    #     tb_dict.update({
    #         'center_loss_cls': point_loss_cls.item(),
    #         'center_pos_num': pos_normalizer.item()
    #     })
    #     return point_loss_cls, tb_dict


    # center class loss, 修改为多层
    # @torchsnooper.snoop()
    def get_center_cls_layer_loss(self, ret_dict, tb_dict=None):
        multi_cls_labels = ret_dict['multi_point_cls_labels']
        multi_cls_preds = ret_dict['multi_layer_cls_preds']
        multi_point_loss_cls = 0
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        for i in range(len(multi_cls_labels)):
            point_cls_labels = multi_cls_labels[i].view(-1)
            point_cls_preds = multi_cls_preds[i].view(-1, self.num_class)

            positives = (point_cls_labels > 0)
            negative_cls_weights = (point_cls_labels == 0) * 1.0
            cls_weights = (negative_cls_weights + 1.0 * positives).float()
            pos_normalizer = positives.sum(dim=0).float()
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)

            one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
            one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
            one_hot_targets = one_hot_targets[..., 1:]

            if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION and (i==(len(multi_cls_labels)-1)):
                centerness_mask = self.generate_center_ness_mask(ret_dict)  # 获得class的centerness mask,只对最后一层
                one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])
                cls_loss_src = loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(point_cls_preds, one_hot_targets)
                cls_loss_src = cls_loss_src * cls_weights.unsqueeze(-1)
            else:
                cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
            point_loss_cls = cls_loss_src.sum()
            multi_point_loss_cls = multi_point_loss_cls + point_loss_cls * loss_weights_dict['multi_scale_cls_weight'][i]

        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        multi_point_loss_cls = multi_point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls.item(),   # 最底层的cls_loss
            'center_pos_num': pos_normalizer.item(),     # 最底层的pos_num
            'multi_loss_cls': multi_point_loss_cls.item()  #各层cls loss加权求和
        })
        # return point_loss_cls, tb_dict
        return multi_point_loss_cls, tb_dict

    # 计算cls_loss,对于最底层,需要计算center_ness_mask
    # @torchsnooper.snoop()
    def generate_center_ness_mask(self, ret_dict):
        pos_mask = ret_dict['center_cls_labels'] > 0
        gt_boxes = ret_dict['center_gt_box_of_fg_points'].clone().detach()  # 最底层的box_gt
        pred_boxes = ret_dict['center_encode_box_preds']  # 最底层的box_preds
        pred_boxes = pred_boxes[pos_mask].clone().detach()

        offset_xyz = pred_boxes[:, 0:3] - gt_boxes[:, 0:3]
        offset_xyz_canical = common_utils.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -gt_boxes[:, 6]).squeeze(dim=1)

        template = gt_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = gt_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])

        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)

        centerness_mask = pos_mask.new_zeros(pos_mask.shape).float()
        centerness_mask[pos_mask] = centerness
        return centerness_mask

    # # @torchsnooper.snoop()
    # def get_center_box_binori_layer_loss(self, tb_dict=None):
    #     pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
    #     point_box_labels = self.forward_ret_dict['center_box_labels']
    #     point_box_preds = self.forward_ret_dict['center_box_preds']
    #
    #     reg_weights = pos_mask.float()
    #     pos_normalizer = pos_mask.sum().float()
    #     reg_weights /= torch.clamp(pos_normalizer, min=1.0)
    #
    #     pred_box_xyzwhl = point_box_preds[:, :6]
    #     label_box_xyzwhl = point_box_labels[:, :6]
    #
    #     point_loss_box_src = self.reg_loss_func(
    #         pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
    #     )
    #     point_loss_xyzwhl = point_loss_box_src.sum()
    #
    #     pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
    #     pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]
    #
    #     label_ori_bin_id = point_box_labels[:, 6]
    #     label_ori_bin_res = point_box_labels[:, 7]
    #     criterion = torch.nn.CrossEntropyLoss(reduction='none')
    #     loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
    #     loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)
    #
    #     label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
    #     pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
    #     loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
    #     loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)
    #
    #     point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
    #     loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
    #     point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
    #     if tb_dict is None:
    #         tb_dict = {}
    #     tb_dict.update({'center_loss_box': point_loss_box.item()})
    #     tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
    #     tb_dict.update({'center_loss_box_ori_cls': loss_ori_cls.item()})
    #     tb_dict.update({'center_loss_box_ori_reg': loss_ori_reg.item()})
    #     return point_loss_box, tb_dict

    # @torchsnooper.snoop()
    # multi scale box loss: (box loss xyzhwl reg loss) + (ori reg loss) + (ori cls loss)
    def get_center_box_binori_layer_loss(self, ret_dict, tb_dict=None):
        multi_cls_labels = ret_dict['multi_point_cls_labels']
        multi_box_labels = ret_dict['multi_point_box_labels']
        multi_box_preds = ret_dict['multi_layer_box_preds']
        multi_point_loss_box = 0
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        # 对各层分别计算box loss
        for i in range(len(multi_cls_labels)):
            pos_mask = multi_cls_labels[i] > 0
            point_box_labels = multi_box_labels[i]
            point_box_preds = multi_box_preds[i]

            reg_weights = pos_mask.float()
            pos_normalizer = pos_mask.sum().float()
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

            pred_box_xyzwhl = point_box_preds[:, :6]
            label_box_xyzwhl = point_box_labels[:, :6]

            point_loss_box_src = self.reg_loss_func(
                pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
            )
            # xyzwhl loss
            point_loss_xyzwhl = point_loss_box_src.sum()

            # ori loss
            pred_ori_bin_id = point_box_preds[:, 6:6+self.box_coder.bin_size]
            pred_ori_bin_res = point_box_preds[:, 6+self.box_coder.bin_size:]

            label_ori_bin_id = point_box_labels[:, 6]
            label_ori_bin_res = point_box_labels[:, 7]
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            loss_ori_cls = criterion(pred_ori_bin_id.contiguous(), label_ori_bin_id.long().contiguous())
            loss_ori_cls = torch.sum(loss_ori_cls * reg_weights)

            label_id_one_hot = F.one_hot(label_ori_bin_id.long().contiguous(), self.box_coder.bin_size)
            pred_ori_bin_res = torch.sum(pred_ori_bin_res * label_id_one_hot.float(), dim=-1)
            loss_ori_reg = F.smooth_l1_loss(pred_ori_bin_res, label_ori_bin_res)
            loss_ori_reg = torch.sum(loss_ori_reg * reg_weights)

            point_loss_box = point_loss_xyzwhl + loss_ori_reg + loss_ori_cls
            multi_point_loss_box = multi_point_loss_box + point_loss_box * loss_weights_dict['multi_scale_box_weight'][i]

        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        multi_point_loss_box = multi_point_loss_box * loss_weights_dict['point_box_weight']

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        tb_dict.update({'multi_loss_box': multi_point_loss_box.item()})
        tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl.item()})
        tb_dict.update({'center_loss_box_ori_cls': loss_ori_cls.item()})
        tb_dict.update({'center_loss_box_ori_reg': loss_ori_reg.item()})
        # return point_loss_box, tb_dict
        return multi_point_loss_box, tb_dict

    def get_center_box_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['center_cls_labels'] > 0
        point_box_labels = ret_dict['center_box_labels']
        point_box_preds = ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss = point_loss_box_src.sum()

        point_loss_box = point_loss
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    # def get_corner_layer_loss(self, tb_dict=None):
    #     pos_mask = self.forward_ret_dict['center_cls_labels'] > 0
    #     gt_boxes = self.forward_ret_dict['center_gt_box_of_fg_points']
    #     pred_boxes = self.forward_ret_dict['point_box_preds']
    #     pred_boxes = pred_boxes[pos_mask]
    #     loss_corner = loss_utils.get_corner_loss_lidar(
    #         pred_boxes[:, 0:7],
    #         gt_boxes[:, 0:7]
    #     )
    #     loss_corner = loss_corner.mean()
    #     loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
    #     if tb_dict is None:
    #         tb_dict = {}
    #     tb_dict.update({'corner_loss_reg': loss_corner.item()})
    #     return loss_corner, tb_dict

    # multi scale corner loss
    def get_corner_layer_loss(self, ret_dict, tb_dict=None):
        multi_cls_labels = ret_dict['multi_point_cls_labels']
        multi_gt_box_of_fg_points = ret_dict['multi_gt_box_of_fg_points']
        # multi_gt_box = self.forward_ret_dict['multi_point_box_labels']
        multi_box_preds = ret_dict['multi_layer_encode_box_preds']  # 解码为框后的box_pred
        multi_loss_corner = 0
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        for i in range(len(multi_cls_labels)-1):
            pos_mask = multi_cls_labels[i] > 0
            gt_boxes = multi_gt_box_of_fg_points[i][pos_mask].clone().detach()
            pred_boxes = multi_box_preds[i]
            pred_boxes = pred_boxes[pos_mask]
            loss_corner = loss_utils.get_corner_loss_lidar(
                pred_boxes[:, 0:7],
                gt_boxes[:, 0:7]
            )
            loss_corner = loss_corner.mean()
            multi_loss_corner = multi_loss_corner + loss_corner*loss_weights_dict['multi_scale_corner_weight'][i]
        # 最底层
        pos_mask = multi_cls_labels[-1] > 0
        gt_boxes = multi_gt_box_of_fg_points[-1].clone().detach()
        pred_boxes = multi_box_preds[-1]
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        multi_loss_corner = multi_loss_corner + loss_corner * loss_weights_dict['multi_scale_corner_weight'][-1]

        loss_corner = loss_corner * loss_weights_dict['corner_weight']
        multi_loss_corner = multi_loss_corner * loss_weights_dict['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner.item()})
        tb_dict.update({'multi_corner_loss_reg': multi_loss_corner.item()})
        # return loss_corner, tb_dict
        return multi_loss_corner, tb_dict

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
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
        """
        if ('gt_boxes' in batch_dict):
            if (batch_dict['gt_boxes'].shape[2]==9):
                self.get_track_label = True
        # center_features = batch_dict['centers_features']
        # center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        # center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        # center_cls_preds_max, _ = center_cls_preds.max(dim=-1)
        # batch_dict['center_cls_scores'] = torch.sigmoid(center_cls_preds_max)

        # 多层检测头 替换之前的单层检测
        batch_size = batch_dict['batch_size']
        cls_preds = []
        box_preds = []
        cls_scores = []
        cls_pred = None
        box_pred = None
        features = batch_dict['multi_features']
        xyzs = batch_dict['multi_xyzs']
        for i in range(len(self.cls_head_modules)):
            feat = features[-1-i]
            xyz = xyzs[-1-i]
            # print("feat: "+str(feat.shape))
            # print("xyz: "+str(xyz.shape))
            cls_feat = feat # [B*n,C1]
            box_feat = feat # [B*n,C2]
            # 除了最底层head,别的层的head都将结果上采样后和当前head层的特征进行拼接
            if i>0:
                cls_pred_up = self.upsample(xyz[:, 1:4].reshape(batch_size, -1, 3).contiguous(),
                                            xyzs[-i][:, 1:4].reshape(batch_size, -1, 3).contiguous(),
                                            cls_pred.reshape(batch_size, -1, cls_pred.shape[1]).permute(0, 2, 1).contiguous()) # B C1 N
                cls_pred_up = cls_pred_up.permute(0, 2, 1).reshape(-1, cls_pred_up.shape[1])  # B*N C1
                cls_feat = torch.cat((cls_feat, cls_pred_up), 1)

                box_pred_up = self.upsample(xyz[:, 1:4].reshape(batch_size, -1, 3).contiguous(),
                                            xyzs[-i][:, 1:4].reshape(batch_size, -1, 3).contiguous(),
                                            box_pred.reshape(batch_size, -1, box_pred.shape[1]).permute(0, 2, 1).contiguous())  # B C2 N
                box_pred_up = box_pred_up.permute(0, 2, 1).reshape(-1, box_pred_up.shape[1])  # B*N C1
                box_feat = torch.cat((box_feat, box_pred_up), 1)
            cls_head = self.cls_head_modules[-1-i] # B*N, C1
            cls_pred = cls_head(cls_feat)
            box_head = self.box_head_modules[-1-i]
            box_pred = box_head(box_feat)
            cls_pred_max, _ = cls_pred.max(dim=-1)
            cls_score = torch.sigmoid(cls_pred_max)
            cls_preds.insert(0, cls_pred)
            box_preds.insert(0, box_pred)
            cls_scores.insert(0, cls_score)

        batch_dict['cls_scores'] = cls_scores
        center_cls_preds = cls_preds[-1]
        center_box_preds = box_preds[-1]
        center_cls_scores = cls_scores[-1]
        batch_dict['center_cls_scores'] = center_cls_scores
        ret_dict = {
                    'multi_layer_cls_preds': cls_preds, # 各层cls_head输出, 从最高层到最低层
                    'multi_layer_box_preds': box_preds, # 各层box_head输出, 从最高层到最低层
                    'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin']
                    }

        # 输出各层cls和box的预测结果,用于场景流层的输入
        batch_dict['multi_layer_cls_preds'] = cls_preds  # 各层cls_head输出, 从最高层到最低层
        batch_dict['multi_layer_box_preds'] = box_preds  # 各层box_head输出, 从最高层到最低层


        if self.training:
            # 获得各个点的label
            targets_dict = self.assign_targets(batch_dict)  # 已改为多层
            ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['multi_point_cls_labels'] = targets_dict['multi_point_cls_labels']
            ret_dict['multi_point_box_labels'] = targets_dict['multi_point_box_labels']
            ret_dict['multi_point_part_labels'] = targets_dict['multi_point_part_labels']
            ret_dict['multi_gt_box_of_fg_points'] = targets_dict['multi_gt_box_of_fg_points']
            ret_dict['center_cls_labels'] = targets_dict['center_cls_labels']
            ret_dict['center_box_labels'] = targets_dict['center_box_labels']
            ret_dict['center_gt_box_of_fg_points'] = targets_dict['center_gt_box_of_fg_points']
            if self.get_track_label:
                ret_dict['multi_track_id_labels'] = targets_dict['multi_point_track_id_labels']
                batch_dict['multi_track_id_labels'] = targets_dict['multi_point_track_id_labels']
                ret_dict['center_track_id_labels'] = targets_dict['center_track_id_labels']
                batch_dict['center_track_id_labels'] = targets_dict['center_track_id_labels']


        # 解码得到预测框
        if not self.training  or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:

            multi_layer_encode_cls_preds = []
            multi_layer_encode_box_preds = []
            for i in range(len(self.cls_head_modules)):
                # print("batch_dict['multi_xyzs'][i][:, 1:4]: " + str(batch_dict['multi_xyzs'][i][:, 1:4].shape))
                # print("cls_preds[i]: "+str(cls_preds[i].shape))
                # print("box_preds[i]: " + str(box_preds[i].shape))
                point_cls_preds_cur_layer, point_box_preds_cur_layer = self.generate_predicted_boxes(
                    points=batch_dict['multi_xyzs'][i][:, 1:4],
                    point_cls_preds=cls_preds[i], point_box_preds=box_preds[i]
                )
                multi_layer_encode_cls_preds.append(point_cls_preds_cur_layer) #各层encode后cls结果,从最高层到最低层
                multi_layer_encode_box_preds.append(point_box_preds_cur_layer) #各层encode后box结果,从最高层到最低层
            batch_dict['multi_layer_box_preds'] = multi_layer_encode_box_preds
            batch_dict['multi_layer_cls_preds'] = multi_layer_encode_cls_preds
            batch_dict['batch_cls_preds'] = multi_layer_encode_cls_preds[-1]
            batch_dict['batch_box_preds'] = multi_layer_encode_box_preds[-1]
            # batch_dict['batch_cls_preds'] = point_cls_preds
            # batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['ctr_batch_idx']
            batch_dict['cls_preds_normalized'] = False

        #     point_cls_preds, point_box_preds = self.generate_predicted_boxes(
        #     points=batch_dict['centers'][:, 1:4],
        #     point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
        # )

            if self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION:
                ret_dict['multi_layer_encode_box_preds'] = multi_layer_encode_box_preds  # 各点预测框
                ret_dict['center_encode_box_preds'] = multi_layer_encode_box_preds[-1]  # 各点预测框
                # ret_dict['center_encode_box_preds'] = point_box_preds

        self.forward_ret_dict = ret_dict  # 用于计算detction loss

        return batch_dict, ret_dict

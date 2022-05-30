import torch
import torch.nn.functional as F

from ...utils import box_coder_utils, box_utils, loss_utils, common_utils
from .point_head_template import PointHeadTemplate
import torchsnooper

class PointHeadBox3DSSD(PointHeadTemplate):
    """
    A simple point-based segmentation head, which are used for PointRCNN.
    Reference Paper: https://arxiv.org/abs/1812.04244
    PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud
    """
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        # 默认情况下不生成每个点的track标签
        self.get_track_label = False

        self.predict_boxes_when_training = predict_boxes_when_training

        target_cfg = self.model_cfg.TARGET_CONFIG
        self.box_coder = getattr(box_coder_utils, target_cfg.BOX_CODER)(
            **target_cfg.BOX_CODER_CONFIG
        )

        self.cls_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=512,
            output_channels=num_class
        )
        self.box_center_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=512,
            output_channels=self.box_coder.code_size
        )
    def get_a(self):
        self.a = self.a + 1
        return self.a

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
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=self.model_cfg.TARGET_CONFIG.GT_EXTRA_WIDTH
        ).view(batch_size, -1, gt_boxes.shape[-1])

        centers = input_dict['centers'].detach()
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert centers.shape.__len__() in [2], 'points.shape=%s' % str(centers.shape)
        targets_dict_center = self.assign_stack_targets(
            points=centers, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=True, ret_track_labels=self.get_track_label
        )
        targets_dict_center['center_gt_box_of_fg_points'] = targets_dict_center['gt_box_of_fg_points']
        targets_dict_center['center_cls_labels'] = targets_dict_center['point_cls_labels']
        targets_dict_center['center_box_labels'] = targets_dict_center['point_box_labels']
        # print("targets_dict_center['center_gt_box_of_fg_points']: \n"+ str(targets_dict_center['center_gt_box_of_fg_points']))
        # print("targets_dict_center['center_cls_labels']: \n"+str(targets_dict_center['center_cls_labels']))
        # print("targets_dict_center['center_box_labels']: \n" + str(targets_dict_center['center_box_labels']))
        targets_dict = targets_dict_center

        return targets_dict

    def get_loss(self, ret_dict, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # point_loss_cls, tb_dict_1 = self.get_cls_layer_loss()
        # point_loss_box, tb_dict_2 = self.get_box_layer_loss()
        pos_mask = ret_dict['center_cls_labels']
        if pos_mask.sum() > 0:
            center_loss_reg, tb_dict_3 = self.get_center_reg_layer_loss(ret_dict)
            center_loss_cls, tb_dict_4 = self.get_center_cls_layer_loss(ret_dict)
            center_loss_box, tb_dict_5 = self.get_center_box_binori_layer_loss(ret_dict)
            corner_loss, tb_dict_6 = self.get_corner_layer_loss(ret_dict)

            # point_loss = point_loss_cls + point_loss_box + center_loss_reg + center_loss_cls + center_loss_box
            point_loss = center_loss_reg + center_loss_cls + center_loss_box + corner_loss
            # print("center_loss_reg %s, center_loss_cls %s, center_loss_box%s, corner_loss %s" %(center_loss_reg, center_loss_cls, center_loss_box, corner_loss) )
            # tb_dict.update(tb_dict_1)
            # tb_dict.update(tb_dict_2)
            tb_dict.update(tb_dict_3)
            tb_dict.update(tb_dict_4)
            tb_dict.update(tb_dict_5)
            tb_dict.update(tb_dict_6)
        else:
            point_loss = torch.tensor(0).cuda(pos_mask.device).float()
            point_loss.requires_grad = True
            tb_dict.update({'center_loss_reg': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_loss_cls': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_pos_num': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_loss_box': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_loss_box_xyzwhl': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_loss_box_ori_cls': torch.tensor(0).cuda(pos_mask.device).float(),
                            'center_loss_box_ori_reg': torch.tensor(0).cuda(pos_mask.device).float(),
                            'corner_loss_reg': torch.tensor(0).cuda(pos_mask.device).float(),
                            })
        print("point loss\n", point_loss.requires_grad)

        return point_loss, tb_dict

    def get_center_reg_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['center_cls_labels'] > 0
        center_box_labels = ret_dict['center_gt_box_of_fg_points'][:, 0:3]
        centers_origin = ret_dict['centers_origin']
        ctr_offsets = ret_dict['ctr_offsets']
        centers_pred = centers_origin + ctr_offsets
        centers_pred = centers_pred[pos_mask][:, 1:4]

        center_loss_box = F.smooth_l1_loss(
            centers_pred, center_box_labels
        )

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_reg': center_loss_box})
        return center_loss_box, tb_dict


    def get_center_cls_layer_loss(self, ret_dict, tb_dict=None):
        point_cls_labels = ret_dict['center_cls_labels'].view(-1)
        point_cls_preds = ret_dict['center_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]

        if self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:
            centerness_mask = self.generate_center_ness_mask(ret_dict)
            one_hot_targets = one_hot_targets * centerness_mask.unsqueeze(-1).repeat(1, one_hot_targets.shape[1])
            cls_loss_src = loss_utils.SigmoidFocalClassificationLoss.sigmoid_cross_entropy_with_logits(point_cls_preds, one_hot_targets)
            cls_loss_src = cls_loss_src * cls_weights.unsqueeze(-1)
        else:
            cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'center_loss_cls': point_loss_cls,
            'center_pos_num': pos_normalizer
        })
        return point_loss_cls, tb_dict

    def generate_center_ness_mask(self, ret_dict):
        pos_mask = ret_dict['center_cls_labels'] > 0
        gt_boxes = ret_dict['center_gt_box_of_fg_points']
        pred_boxes = ret_dict['point_box_preds']
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

    # @torchsnooper.snoop()
    def get_center_box_binori_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['center_cls_labels'] > 0
        point_box_labels = ret_dict['center_box_labels']
        point_box_preds = ret_dict['center_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        pred_box_xyzwhl = point_box_preds[:, :6]
        label_box_xyzwhl = point_box_labels[:, :6]

        point_loss_box_src = self.reg_loss_func(
            pred_box_xyzwhl[None, ...], label_box_xyzwhl[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_xyzwhl = point_loss_box_src.sum()

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
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'center_loss_box': point_loss_box})
        tb_dict.update({'center_loss_box_xyzwhl': point_loss_xyzwhl})
        tb_dict.update({'center_loss_box_ori_cls': loss_ori_cls})
        tb_dict.update({'center_loss_box_ori_reg': loss_ori_reg})
        return point_loss_box, tb_dict

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
        tb_dict.update({'center_loss_box': point_loss_box})
        return point_loss_box, tb_dict

    def get_corner_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['center_cls_labels'] > 0
        gt_boxes = ret_dict['center_gt_box_of_fg_points']
        pred_boxes = ret_dict['point_box_preds']
        pred_boxes = pred_boxes[pos_mask]
        loss_corner = loss_utils.get_corner_loss_lidar(
            pred_boxes[:, 0:7],
            gt_boxes[:, 0:7]
        )
        loss_corner = loss_corner.mean()
        loss_corner = loss_corner * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['corner_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'corner_loss_reg': loss_corner})
        return loss_corner, tb_dict

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
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """

        # if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
        #     point_features = batch_dict['point_features_before_fusion']
        # else:
        #     point_features = batch_dict['point_features']
        # point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)
        # point_box_preds = self.box_layers(point_features)  # (total_points, box_code_size)
        #
        # point_cls_preds_max, _ = point_cls_preds.max(dim=-1)
        # batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)
        if ('gt_boxes' in batch_dict):
            if batch_dict['gt_boxes'].shape[2]==9:
                self.get_track_label = True
        center_features = batch_dict['centers_features']
        center_cls_preds = self.cls_center_layers(center_features)  # (total_centers, num_class)
        center_box_preds = self.box_center_layers(center_features)  # (total_centers, box_code_size)
        center_cls_preds_max, _ = center_cls_preds.max(dim=-1)
        batch_dict['center_cls_scores'] = torch.sigmoid(center_cls_preds_max)

        # ret_dict = {'point_cls_preds': point_cls_preds,
        #             'point_box_preds': point_box_preds,
        #             'center_cls_preds': center_cls_preds,
        #             'center_box_preds': center_box_preds,
        #             'ctr_offsets': batch_dict['ctr_offsets'],
        #             'centers': batch_dict['centers'],
        #             'centers_origin': batch_dict['centers_origin']}
        ret_dict = {'center_cls_preds': center_cls_preds,
                    'center_box_preds': center_box_preds,
                    'ctr_offsets': batch_dict['ctr_offsets'],
                    'centers': batch_dict['centers'],
                    'centers_origin': batch_dict['centers_origin']}
        # if self.training:
        # eval需要计算场景流的loss,所以需要在eval阶段也获取det label
        if ('gt_boxes' in batch_dict):
            targets_dict = self.assign_targets(batch_dict)
            # ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            # ret_dict['point_box_labels'] = targets_dict['point_box_labels']
            ret_dict['center_cls_labels'] = targets_dict['center_cls_labels']
            ret_dict['center_box_labels'] = targets_dict['center_box_labels']
            # print("center box label: " + str(ret_dict['center_box_labels']))
            ret_dict['center_gt_box_of_fg_points'] = targets_dict['center_gt_box_of_fg_points']


        if not self.training or self.predict_boxes_when_training or \
                self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION or \
                self.model_cfg.LOSS_CONFIG.CENTERNESS_REGULARIZATION:

            # point_cls_preds, point_box_preds = self.generate_predicted_boxes(
            #     points=batch_dict['point_coords'][:, 1:4],
            #     point_cls_preds=point_cls_preds, point_box_preds=point_box_preds
            # )

            point_cls_preds, point_box_preds = self.generate_predicted_boxes(
                    points=batch_dict['centers'][:, 1:4],
                    point_cls_preds=center_cls_preds, point_box_preds=center_box_preds
                )
            batch_dict['batch_cls_preds'] = point_cls_preds
            batch_dict['batch_box_preds'] = point_box_preds
            batch_dict['batch_index'] = batch_dict['ctr_batch_idx']  #
            batch_dict['cls_preds_normalized'] = False

            if self.model_cfg.LOSS_CONFIG.CORNER_LOSS_REGULARIZATION:
                ret_dict['point_box_preds'] = point_box_preds

            # cls_scores, cls_labels = torch.max(point_cls_preds, dim=1)
            # batch_size = batch_dict['batch_size']
            # batch_dict['batch_pred_labels'] = cls_labels.view(batch_size, -1) + 1
            # batch_dict['batch_cls_preds'] = cls_scores.view(batch_size, -1).unsqueeze(-1)
            # batch_dict['batch_box_preds'] = point_box_preds.view(batch_size, -1, 7)
            # batch_dict['cls_preds_normalized'] = False
            # batch_dict.pop('batch_index', None)

        self.forward_ret_dict = ret_dict
        return batch_dict, ret_dict


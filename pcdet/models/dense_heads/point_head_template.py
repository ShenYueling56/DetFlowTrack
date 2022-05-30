import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils
torch.set_printoptions(edgeitems=100000, threshold=100000)
import torchsnooper

class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)

        # self.forward_ret_dict = None
        self.a = 0


    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_type = losses_cfg.get('LOSS_REG', None)
        if reg_loss_type == 'smooth-l1':
            self.reg_loss_func = F.smooth_l1_loss
        elif reg_loss_type == 'l1':
            self.reg_loss_func = F.l1_loss
        elif reg_loss_type == 'WeightedSmoothL1Loss':
            self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
            )
        else:
            self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    # @torchsnooper.snoop()
    def multi_scale_assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, ret_track_labels=False, central_radius=2.0):
        """
        获得各点label
        Args:
            points: (N1 + N2 + N3 + ..., 4) [batch_id, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)
            gt_box_of_fg_points: (B*N, 7): 各点所在的gt_box_lidar

        """

        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        if ret_track_labels:
            assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 9, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 9, \
                'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        else:
            assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
                'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'

        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        gt_boxes_of_fg_points = []
        # 对各个batch分别操作
        for k in range(batch_size):
            # 获取当前batch的points_single
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())  # 当前batch的cls_labels:point_cls_labels_single
            # 获得各点所在框的id,id=0表示点不在框内
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            # 如果cls_label存在ignore,对ignore范围内的点cls_label赋值为-1
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag # 框内点标识符
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0) #ignore范围内点标识符
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            # print("box_idxs_of_pts[fg_flag] shape: " + str(box_idxs_of_pts[fg_flag].shape)) #框内点的box idx
            import copy
            box_idxs_of_pts_copy = copy.deepcopy(box_idxs_of_pts) # 将idx=-1的修改为0
            # print("box_idxs_of_pts: \n"+str(box_idxs_of_pts))
            box_idxs_of_pts_copy = torch.clamp(box_idxs_of_pts_copy,0)
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts_copy][:, :8]   # 各个点的box label,框外点的box_label为box[0]
            # print("fg_flag shape: "+str(fg_flag.shape))
            # print("point_cls_labels_single shape: "+str(point_cls_labels_single.shape))
            # print("gt_box_of_fg_points shape: "+str(gt_box_of_fg_points[fg_flag].shape))
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[fg_flag][:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            gt_boxes_of_fg_points.append(gt_box_of_fg_points)

            if ret_box_labels and gt_box_of_fg_points[fg_flag].shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[fg_flag][:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[fg_flag][:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                # print("point_box_labels_single max: " + str(torch.max(point_box_labels_single, 0)[0]))  # 16384*8
                # print("point_box_labels_single min: " + str(torch.min(point_box_labels_single, 0)[0]))  # 16384*8
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[fg_flag][:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[fg_flag][:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[fg_flag][:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,   # encode后的box_label
            'point_part_labels': point_part_labels,
            'box_idxs_of_pts': box_idxs_of_pts,  # 各点对应的gt_boxes7维数据+cls(一共8维)
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
        }

        return targets_dict

    # @torchsnooper.snoop()
    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, ret_track_labels=False, central_radius=2.0):
        """
        获得各点label
        Args:
            points: (N1 + N2 + N3 + ..., 4) [batch_id, x, y, z]
            gt_boxes: (B, M, 8) or (B, M, 9)
            extend_gt_boxes: [B, M, 8] or [B, M, 9]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)
            gt_box_of_fg_points: (B*N, 7): 各点所在的gt_box_lidar

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        if ret_track_labels:
            assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 9, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 9, \
                'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        else:
            assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
            assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None

        gt_boxes_of_fg_points = []
        # 对各个batch分别操作
        for k in range(batch_size):
            # 获取当前batch的points_single
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())  # 当前batch的cls_labels:point_cls_labels_single
            # 获得各点所在框的id,id=0表示点不在框内
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            # 如果cls_label存在ignore,对ignore范围内的点cls_label赋值为-1
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag # 框内点标识符
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0) #ignore范围内点标识符
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            # class label
            # print("box_idxs_of_pts[fg_flag] shape: " + str(box_idxs_of_pts[fg_flag].shape)) #框内点的box idx
            if gt_boxes[k].shape[0] == 0:
                continue
            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts][[fg_flag]][:, :8]  # 各个点的box label,框外点的box_label为box[0]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, 7].long()
            point_cls_labels[bs_mask] = point_cls_labels_single
            gt_boxes_of_fg_points.append(gt_box_of_fg_points)

            # box label
            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )

                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            # part label
            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single
        if len(gt_boxes_of_fg_points) != 0:
            gt_boxes_of_fg_points = torch.cat(gt_boxes_of_fg_points, dim=0)
        else:
            gt_boxes_of_fg_points = torch.empty(0)
        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,   # encode后的box_label
            'point_part_labels': point_part_labels,
            'box_idxs_of_pts': box_idxs_of_pts,  # 各点对应的gt_boxes7维数据
            'gt_box_of_fg_points': gt_boxes_of_fg_points,
        }
        return targets_dict

    def get_cls_layer_loss(self, ret_dict, tb_dict=None):
        point_cls_labels = ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = ret_dict['point_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = ret_dict['point_part_labels']
        point_part_preds = ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, ret_dict, tb_dict=None):
        pos_mask = ret_dict['point_cls_labels'] > 0
        point_box_labels = ret_dict['point_box_labels']
        point_box_preds = ret_dict['point_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict

    # @torchsnooper.snoop()
    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError

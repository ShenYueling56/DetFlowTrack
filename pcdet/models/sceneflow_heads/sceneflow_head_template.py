import torch
import torch.nn as nn
import torch.nn.functional as F

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils, box_utils
torch.set_printoptions(edgeitems=100000, threshold=100000)
import torchsnooper

class SceneflowHeadTemplate(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.flow_model_cfg = model_cfg
        self.flow_forward_ret_dict = None


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
    def assign_stack_targets(self, points1, gt_boxes1, points2, gt_boxes2):
        """
        获得各点label
        Args:
            points1: (N1 + N2 + N3 + ..., 4) [batch_id, x, y, z]
            gt_boxes1: (B, M, 8)
            point_track_id_labels: (M, 1)

        Returns:
            point_sceneflow_labels: B 3 M
        """
        assert len(points1.shape) == 2 and points1.shape[1] == 4, 'points.shape=%s' % str(points1.shape)
        assert len(points2.shape) == 2 and points2.shape[1] == 4, 'points.shape=%s' % str(points2.shape)
        assert len(gt_boxes1.shape) == 3 and gt_boxes1.shape[2] == 9, 'gt_boxes.shape=%s' % str(gt_boxes1.shape)
        assert len(gt_boxes2.shape) == 3 and gt_boxes2.shape[2] == 9, 'gt_boxes.shape=%s' % str(gt_boxes2.shape)

        batch_size = gt_boxes1.shape[0]
        bs_idx1 = points1[:, 0]
        bs_idx2 = points2[:, 0]
        point_sceneflow_labels = gt_boxes1.new_zeros((points1.shape[0], 3))
        # 标记点是否在框内
        point_inbox_labels = gt_boxes1.new_zeros((points1.shape[0])).long()
        # 对各个batch分别操作
        for k in range(batch_size):
            # 获取当前batch的points_single
            bs_mask1 = (bs_idx1 == k)
            bs_mask2 = (bs_idx2 == k)
            points_single1 = points1[bs_mask1][:, 1:4]
            points_single2 = points2[bs_mask2][:, 1:4]

            gt_boxes_single1 = gt_boxes1[k]
            gt_boxes_single2 = gt_boxes2[k]
            corners_lidar1 = box_utils.boxes_to_corners_3d(gt_boxes_single1[:, 0:7].reshape(-1, 7))  # 各个框的顶点
            point_sceneflow_labels_single = point_sceneflow_labels.new_zeros(bs_mask1.sum(), 3)  # 当前batch的sceneflow_labels_single
            point_inbox_labels_single = point_inbox_labels.new_zeros(bs_mask1.sum()).long()
            # 获得pc1各个点所在框的序列号
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single1.unsqueeze(dim=0), gt_boxes1[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            # 去掉用于补全的全0框
            box_all_zero_flag1 = (gt_boxes_single1[:, 0] == 0) & (gt_boxes_single1[:, 1] == 0) & (
                    gt_boxes_single1[:, 2] == 0)
            box_all_zero_flag2 = (gt_boxes_single2[:, 0] == 0) & (gt_boxes_single2[:, 1] == 0) & (
                        gt_boxes_single2[:, 2] == 0)
            gt_boxes_single1 = gt_boxes_single1[~box_all_zero_flag1]
            gt_boxes_single2 = gt_boxes_single2[~box_all_zero_flag2]

            for box_index1 in range(gt_boxes_single1.shape[0]):
                box_track_id = gt_boxes_single1[box_index1, 8].long()
                box_flag2 = (gt_boxes_single2[:, 8].long())==box_track_id
                if box_flag2.sum() > 0:
                    box_pos1 = gt_boxes_single1[box_index1, 0:3].view(1, 1, 3)
                    box_pos2 = gt_boxes_single2[box_flag2][:, 0:3].view(1, 1, 3)
                    box_ry1 = gt_boxes_single1[box_index1, 6].view(1)
                    box_ry2 = gt_boxes_single2[box_flag2][:, 6].view(1)

                    # 找出框1内的点,并对框1内的点做变换,变换到框2的坐标系下
                    inbox_flag1 = (box_idxs_of_pts == box_index1)
                    points_inbox_single1 = points_single1[inbox_flag1]

                    import copy
                    points_inbox_single12 = copy.deepcopy(points_inbox_single1)
                    points_inbox_single12 = points_inbox_single12 - box_pos1
                    points_inbox_single12 = common_utils.rotate_points_along_z(points_inbox_single12, -box_ry1+box_ry2)
                    points_inbox_single12 = points_inbox_single12 + box_pos2
                    # 得到第一帧点的场景流真实值
                    point_sceneflow_labels_single[inbox_flag1] = points_inbox_single12 - points_inbox_single1
                    point_inbox_labels_single[inbox_flag1] = 1
            point_sceneflow_labels[bs_mask1] = point_sceneflow_labels_single
            point_inbox_labels[bs_mask1] = point_inbox_labels_single

        point_sceneflow_labels = point_sceneflow_labels.reshape(batch_size, -1, 3).permute(0, 2, 1).contiguous() # B 3 N
        point_inbox_labels = point_inbox_labels.reshape(batch_size, -1).contiguous()  # B N
        targets_dict = {
            'point_sceneflow_labels': point_sceneflow_labels,
            'point_inbox_labels': point_inbox_labels,
        }
        return targets_dict


    # 获取场景流的loss
    # @torchsnooper.snoop()
    def get_loss(self, ):

        return

    # @torchsnooper.snoop()
    # 进行两帧框的数据关联

    def forward(self, **kwargs):
        raise NotImplementedError

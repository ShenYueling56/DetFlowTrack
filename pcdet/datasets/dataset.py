from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from pcdet.utils import common_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None, total_gpus=1):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        print(dataset_cfg)
        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training  else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        print("voxel size 00 \n", self.voxel_size)
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.total_gpus = total_gpus

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError

    def prepare_data(self, data_dict, aug_dict=None):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            # 各个gt box类别是否在class_names中
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            # db数据增强和global数据增强,对于track添加仅对框的平移和旋转的数据增强
            data_dict, aug_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                },
                aug_dict=aug_dict,
            )

        # 获取待检测类别的gt boxes
        if data_dict.get('gt_boxes', None) is not None:
            # 根据类名字对gt boxs进行删选
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            if(data_dict.get('track_id', None) is not None):
                data_dict['track_id'] = data_dict['track_id'][selected]
                gt_boxes = np.concatenate(( data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32), data_dict['track_id'].reshape(-1,1)), axis=1)
                data_dict.pop('track_id')
            else:
                gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        # 特征编码 xyz intensity
        data_dict = self.point_feature_encoder.forward(data_dict)
        # 删除范围外的框,降采样,乱序
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        data_dict.pop('gt_names', None)

        return data_dict, aug_dict

    def collate_batch(self, batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        ### 末尾的batch不等于batch_size_per_gpu * total_gpus
        batch_size = len(batch_list)

        ret = {}
        batch_size_per_gpu = int(batch_size/(self.total_gpus))
        if batch_size_per_gpu == 0:
            batch_size_per_gpu = batch_size
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'frame1_voxels', 'frame2_voxels', 'frame1_voxel_num_points', 'frame2_voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'frame1_points', 'frame2_points', 'frame1_voxel_coords', 'frame2_voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        batch_index = int(i%batch_size_per_gpu)
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=batch_index)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'frame1_gt_boxes', 'frame2_gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                # elif key in ['frame1_voxels', 'frame2_voxels']:
                #     # print("dataset cfg\n")
                #     # print(self.dataset_cfg.DATA_PROCESSOR[2])
                #     # print(self.dataset_cfg.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS[self.mode])
                #     # print(self.dataset_cfg.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL)
                #     batch_voxels = np.zeros((batch_size, self.dataset_cfg.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS[self.mode], self.dataset_cfg.DATA_PROCESSOR[2].MAX_POINTS_PER_VOXEL, val[0].shape[-1]), dtype=np.float32)
                #     for k in range(batch_size):
                #         batch_voxels[k, :val[k].__len__(), :] = val[k]
                #     ret[key] = batch_voxels
                # elif key in ['frame1_voxel_coords', 'frame2_voxel_coords']:
                #     batch_coords = np.zeros(shape=(batch_size, self.dataset_cfg.DATA_PROCESSOR[2].MAX_NUMBER_OF_VOXELS[self.mode], 3), dtype=np.int32)
                #     for k in range(batch_size):
                #         batch_coords[k, :val[k].__len__(), :] = val[k]
                #     ret[key] = batch_coords
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        ret['batch_size'] = batch_size_per_gpu
        return ret

# -*- coding: utf-8 -*-
# @File  : kitti_tracking.py
# @Author: syl
# @Date  : 2020/8/1
# @Desc  :
import pickle
import math
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
import copy
import os
import random
import sys
import yaml
from easydict import EasyDict
import mayavi.mlab as mlab

import torch
from torch.utils.data import Dataset, DataLoader

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, tracking_calibration_kitti, tracking_kitti, common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from pcdet.utils.visual_utils import visualize_utils as V

iou_threshold = 0.2
last_last_track_id = []
last_track_id = []
last_match2 = []
first_frame = True
last_seq_id = 15
last_seq_idx = 16

annos_file = []
np.set_printoptions(threshold=np.inf)

MAX_DIS = 0.5
MAX_ANGLE = 18
NAME2ID = {
    'Pedestrian': 1,
    'Car': 2,
    'Cyclist': 3
}
det_none = False
show = False #是否可视化
MAX_MISS_TIME = 3
MIN_APPEAR_LIFE = 3

DEL_TH = 0.2
OUTPUT_TH = 0.5
training_frame_lists = [152, 445, 231, 142, 312, 295, 268, 798, 388, 801,
                        292, 371, 76, 338, 104, 374, 207, 143, 337, 1057, 835, ]
testing_frame_lists = [463, 145, 241, 255, 419, 807, 112, 213, 163, 347,
                       1174, 772, 692, 150, 848, 699, 508, 303, 178, 402,
                       171, 201, 434, 428, 314, 174, 168, 83, 173]

end_frame_lists = {
    "train": training_frame_lists,
    "val": training_frame_lists,
    "test": testing_frame_lists
}

class KittiTracking(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, total_gpus=1):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger, total_gpus=total_gpus
        )  # 对继承自父类的属性进行初始化

        # 设置路径
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        # 获取idx
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.seq_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        # 加载info
        self.kitti_infos = []
        self.include_kitti_data(self.mode)

        print("self.info len: %s" %str(len(self.kitti_infos)))
        # 设置模式:det/track
        self.only_det = dataset_cfg.ONLY_DET
        if self.only_det:
            print("Detection Mode")
        else:
            print("Tracking Mode")

    def __len__(self):
        # 这个函数也必须要写，它返回的是数据集的长度
        return len(self.kitti_infos)

    # 使用__getitem__()对数据进行预处理并返回想要的信息
    def __getitem__(self, index):
        if self.only_det:
            sample = self.get_det_sample(index)
        else:
            sample = self.get_sceneflow_sample(index)
        return sample


    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            print("info_path: " + str(info_path))
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def get_lidar(self, seq_idx, frame_idx):
        # lidar_file = self.root_split_path / 'rmground_points' / ('%04d' % seq_idx) / ('%06d.bin' % frame_idx)
        lidar_file = self.root_split_path / 'velodyne' / ('%04d' % seq_idx) / ('%06d.bin' % frame_idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, seq_idx, frame_idx):
        img_file = self.root_split_path / 'image_02' / ('%s' % seq_idx) / ('%06d.png' % frame_idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def show_image(self, seq_idx, frame_idx):
        img_file = self.root_split_path / 'image_02' / ('%04d' % seq_idx) / ('%06d.png' % frame_idx)
        print("img file: ", img_file)
        assert img_file.exists()
        img = io.imread(img_file)
        io.imshow(img)


    def get_label(self, seq_idx):
        label_file = self.root_split_path / 'label_02' / ('%04d.txt' % seq_idx)
        print(label_file)
        assert label_file.exists()
        return tracking_kitti.get_objects_from_label(label_file)

    def get_calib(self, seq_idx):
        calib_file = self.root_split_path / 'calib' / ('%04d.txt' % seq_idx)
        # print(calib_file)
        assert calib_file.exists()
        return tracking_calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, seq_idx, frame_idx):
        plane_file = self.root_split_path / 'planes' / ('%04d' % seq_idx) / ('%06d.txt' % frame_idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    def get_oxts_seq(self, seq_idx):
        oxts_file = self.root_split_path / 'oxts' / ('%s.txt' % seq_idx)
        oxts = open(oxts_file, 'r')
        oxts_seq = oxts.readlines()
        return oxts_seq

    def get_pos(self, oxts_seq, id):
        oxt = oxts_seq[id].strip().split(' ')
        lat = float(oxt[0])
        lon = float(oxt[1])
        alt = float(oxt[2])
        pos_x, pos_y = self.proj_trans1(lon, lat)
        pos = np.array([pos_x, pos_y, alt])
        rad = np.array([x for x in map(float, oxt[3:6])])
        return pos, rad

    def proj_trans1(self, lon, lat):
        import pyproj
        # p3 = pyproj.Proj("epsg:4326")
        p1 = pyproj.Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=False)
        p2 = pyproj.Proj("epsg:3857")
        x1, y1 = p1(lon, lat)
        x2, y2 = pyproj.transform(p1, p2, x1, y1, radians=True)
        return x2, y2

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib, threshold=5):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 5, pts_img[:, 0] < img_shape[1]-5)
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 5, pts_img[:, 1] < img_shape[0]-5)
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        self.split_dir = self.root_path / 'ImageSets' / self.split
        self.seq_id_list = [int(x[:-4]) for x in os.listdir(self.split_dir)]

    # 获取det sample 都是在点云坐标系下
    def get_det_sample(self, index):
        """
        根据序列号获取一帧的点云坐标系下的点云和标签
        """
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
        info = copy.deepcopy(self.kitti_infos[index][0])
        img_shape = info['image']['image_shape']
        seq_idx = info['idx_info']['seq_idx']
        frame_idx = info['idx_info']['frame_idx']

        # get points from file
        points = self.get_lidar(seq_idx, frame_idx)
        calib = self.get_calib(seq_idx)

        # remove points outside fov
        pts_rect = calib.lidar_to_rect(points[:, :3])
        pts_valid_flag = self.get_fov_flag(pts_rect, img_shape, calib)
        points = points[pts_valid_flag]

        input_dict = {
            'points': points,
            'seq_id': seq_idx,
            'frame_id': frame_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

            road_plane = self.get_road_plane(seq_idx, frame_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict, _ = self.prepare_data(data_dict=input_dict)
        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)
        data_dict['image_shape'] = img_shape

        # 可视化点云和框的标签 和图片
        # self.show_image(seq_idx, frame_idx)
        # V.draw_scenes(data_dict['points'], gt_boxes=data_dict['gt_boxes'])
        # mlab.show(stop=True)
        # plt.show()

        return data_dict


    # 获取sceneflow sample
    def get_sceneflow_sample(self, index):
        # 返回值
        # data_dict:
        # points: N,7
        # seq_id: int
        # frame_id: int
        # gt_boxes: N, 9 [gt_boxes(7 axis), class_index, track_id]
        # use_lead_xyz
        # image_shape
        # print("tracking mode")
        def get_one_scene_sample(info, aug_dict):
            img_shape = info['image']['image_shape']
            seq_idx = info['idx_info']['seq_idx']
            frame_idx = info['idx_info']['frame_idx']
            # print("seq%d frame%d"%(seq_idx,frame_idx))

            # get points from file
            points = self.get_lidar(seq_idx, frame_idx)
            calib = self.get_calib(seq_idx)

            # remove points outside fov
            pts_rect = calib.lidar_to_rect(points[:, :3])
            pts_valid_flag = self.get_fov_flag(pts_rect, img_shape, calib, threshold=20)
            points = points[pts_valid_flag]


            input_dict= {
                'points': points,
                'seq_id': seq_idx,
                'frame_id': frame_idx,
                'calib': calib,
            }

            if 'annos' in info:
                annos = info['annos']
                annos = common_utils.drop_info_with_name(annos, name='DontCare')
                loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
                gt_names = annos['name']
                gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
                track_id = annos['track_id']

                input_dict.update({
                    'gt_names': gt_names,
                    'gt_boxes': gt_boxes_lidar,
                    'track_id': track_id,
                })
                road_plane = self.get_road_plane(seq_idx, frame_idx)
                if road_plane is not None:
                    input_dict['road_plane'] = road_plane

            # V.draw_scenes(points=points, gt_boxes=gt_boxes_lidar)
            # mlab.show(stop=True)
            data_dict, aug_dict = self.prepare_data(data_dict=input_dict, aug_dict=aug_dict)

            data_dict['image_shape'] = img_shape
            return data_dict, aug_dict

        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)
        info = copy.deepcopy(self.kitti_infos[index])
        info1 = info[0]
        info2 = info[1]
        # init aug dict for first frame
        aug_dict = {
            'flip_enable': -1,
            'world_rot_enable': -1,
            'world_rot_angle': 0,
            'world_scale_enable': -1,
            'world_scale': 1,
            'gt_sample': -1,
            'gt_sample_dict': None,
        }
        # get first frame sample
        data_dict1, aug_dict = get_one_scene_sample(info1, aug_dict)
        # get second frame sample
        data_dict2, aug_dict = get_one_scene_sample(info2, aug_dict)
        data_dict_all = {}
        for key, val in data_dict1.items():
            if key == 'calib':
                continue
            data_dict_all["frame1_" + key] = val
        for key, val in data_dict2.items():
            if key == 'calib':
                continue
            data_dict_all["frame2_" + key] = val
        if self.training and ((len(data_dict1['gt_boxes']) == 0) or (len(data_dict2['gt_boxes']) == 0)):
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        # V.draw_two_scenes(points=data_dict1['points'], ref_points=data_dict2['points'], gt_boxes=data_dict1['gt_boxes'][:, :7], ref_boxes=data_dict2['gt_boxes'][:, :7])
        # mlab.show(stop=True)

        return data_dict_all

    def get_infos(self, delta_frame=1, num_workers=4, has_label=True, count_inside_pts=True, seq_id_list=None, eval_aug=False):
        def process_single_scene(seq_idx, sample_idx, seq_objects, seq_calib, seq_otxs, f_label=None, lidar_file=None):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            # todo: 将随机平移和旋转作用于框内的点以及lable,并且重写点云文件和真实值文件.对框内点和gt_box的操作参考data_augmentor文件
            info = {}
            pc_info = {'num_features': 4}
            info['point_cloud'] = pc_info
            image_info = {'image_shape': self.get_image_shape(seq_idx, sample_idx)}
            info['image'] = image_info
            idx_info = {'seq_idx': int(seq_idx), 'frame_idx': int(sample_idx)}
            info['idx_info'] = idx_info

            calib = seq_calib
            # P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            # R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            # R0_4x4[3, 3] = 1.
            # R0_4x4[:3, :3] = calib.R0
            # V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            # I2V_4x4 = np.concatenate([calib.I2V, np.array([[0., 0., 0., 1.]])], axis=0)
            # calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4, 'Tr_imu_to_velo': I2V_4x4}
            # info['calib'] = calib
            obj_list = None
            if has_label:
                ### 变换前
                obj_list = seq_objects[sample_idx]
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # h w l(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)
                annotations['track_id'] = np.array([obj.track_idx for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                object_mask = [obj.cls_type != 'DontCare' for obj in obj_list]
                num_gt = len(np.array([obj.cls_type for obj in obj_list]))
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                # annotations['index'] = np.array(index, dtype=np.int32)



                # gt boxes in lidar
                loc = annotations['location'][object_mask]
                dims = annotations['dimensions'][object_mask]
                rots = annotations['rotation_y'][object_mask]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar
                points = self.get_lidar(int(seq_idx), sample_idx)
                if f_label is not None:
                    print("transform in seq %d, frame%d" % (int(seq_idx), int(sample_idx)))
                    ###人为加上两帧之间的剧烈运动
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar.reshape(-1, 7))  # 各个框的顶点
                    # 逐个框进行操作
                    for i in range(gt_boxes_lidar.shape[0]):
                        # 随机获取平移量
                        displacement = np.random.randn(3) * MAX_DIS
                        displacement[2] = 0
                        # 判断平移后的框是否和原有的框有重合,如何有那不做这次变换
                        new_gt_box_lidar = copy.deepcopy(gt_boxes_lidar[i])
                        new_gt_box_lidar[0:3] = new_gt_box_lidar[0:3] + displacement
                        new_enlarged_gt_box_lidar = copy.deepcopy(new_gt_box_lidar)
                        new_enlarged_gt_box_lidar[3] += 0.5
                        new_enlarged_gt_box_lidar[4] += 0.5
                        new_enlarged_gt_box_lidar[5] += 2

                        if i > 1:
                            iou = box_utils.boxes3d_nearest_bev_iou(new_enlarged_gt_box_lidar.reshape(1, 7), gt_boxes_lidar[:i, :])
                            if iou.max() > 1e-8:
                                continue
                        if i + 1 < gt_boxes_lidar.shape[0]:
                            iou = box_utils.boxes3d_nearest_bev_iou(new_enlarged_gt_box_lidar.reshape(1, 7), gt_boxes_lidar[i + 1:, :])
                            if iou.max() > 1e-8:
                                continue
                        # 对框做平移变换
                        gt_boxes_lidar[i, 0:3] = gt_boxes_lidar[i, 0:3] + displacement
                        # 对框内点做平移变换
                        pt_mask_flag = box_utils.in_hull(points[:, :3], corners_lidar[i])
                        points[pt_mask_flag, :3] = points[pt_mask_flag, :3] + displacement
                        # 去掉原本点云中在平移后的框范围内的点
                        enlarged_corners_lidar = box_utils.boxes_to_corners_3d(
                            new_enlarged_gt_box_lidar.reshape(-1, 7))  # 各个框的顶点
                        pt_del_mask_flag = box_utils.in_hull(points[:, :3], enlarged_corners_lidar[0])
                        points = points[(pt_del_mask_flag < 1) | (pt_mask_flag)]

                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar.reshape(-1, 7))  # 各个框的顶点

                    if MAX_ANGLE > 0:
                        for i in range(gt_boxes_lidar.shape[0]):

                            # 随机获取旋转量
                            rot_angle = np.random.randn(1) * np.pi / MAX_ANGLE
                            # 将旋转角作用于框
                            new_gt_box_lidar = copy.deepcopy(gt_boxes_lidar[i])
                            new_gt_box_lidar[6] += rot_angle
                            gt_boxes_lidar[i] = new_gt_box_lidar
                            # 将旋转角作用于框内点,需要将点首先转换到以框为中心的局部坐标系下,再旋转点
                            pt_mask_flag = box_utils.in_hull(points[:, :3], corners_lidar[i])
                            new_points = points[pt_mask_flag, :3] - new_gt_box_lidar[:3]
                            new_points = common_utils.rotate_points_along_z(new_points[np.newaxis, :, :3], np.array(rot_angle))[
                                0]
                            points[pt_mask_flag, :3] = new_points + new_gt_box_lidar[:3]

                    # 更新annotations中的gt_boxes_lidar,location,bbox,
                    annotations['gt_boxes_lidar'] = gt_boxes_lidar
                    loc_lidar = copy.deepcopy(gt_boxes_lidar[:, 0:3])
                    loc_lidar[:, 2] -= h[:, 0] / 2
                    loc_rect = calib.lidar_to_rect(loc_lidar)
                    annotations['location'][object_mask] = loc_rect
                    annotations['rotation_y'][object_mask] = -gt_boxes_lidar[: ,6] - np.pi / 2
                    gt_boxes_rect = np.concatenate([annotations['location'][object_mask], annotations['dimensions'][object_mask], annotations['rotation_y'][object_mask].reshape(-1,1)], axis=1)
                    gt_boxes2d_image = box_utils.boxes3d_kitti_camera_to_imageboxes(gt_boxes_rect, calib)
                    annotations['bbox'][object_mask] = gt_boxes2d_image

                    # 更新对应帧的obj
                    for i, obj in enumerate(obj_list):
                        obj = obj.set_after_trans(annotations['location'][i], annotations['rotation_y'][i], annotations['bbox'][i])
                        obj_list[i] = obj
                    seq_objects[sample_idx] = obj_list
                    # 每两帧里的后一帧写入label文件
                    if f_label is not None:
                        for i in range(annotations['track_id'].shape[0]):
                            kitti_str = "%d %d %s %d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f \n" \
                                        % (sample_idx, annotations['track_id'][i], annotations['name'][i], annotations['truncated'][i], annotations['occluded'][i], annotations['alpha'][i], annotations['bbox'][i][0],
                                           annotations['bbox'][i][1], annotations['bbox'][i][2], annotations['bbox'][i][3], annotations['dimensions'][i][1], annotations['dimensions'][i][2], annotations['dimensions'][i][0],
                                           annotations['location'][i][0], annotations['location'][i][1], annotations['location'][i][2],
                                           annotations['rotation_y'][i])
                            f_label.write(kitti_str)

                    # 将两帧里的后一帧的点云写入点云文件
                    if lidar_file is not None:
                        f_lidar = open(lidar_file + '%06d.bin' % int(sample_idx), 'wb')
                        f_lidar.write(points)
                        f_lidar.close()

                # V.draw_scenes(points, gt_boxes=gt_boxes_lidar)
                # mlab.show(stop=True)
                # print("seq frame name\n")
                # print(seq_idx, " ", sample_idx, " ", annotations['name'])

                info['annos'] = annotations
                # if count_inside_pts:
                #     points = self.get_lidar(int(seq_idx), sample_idx)
                #     calib = seq_calib
                #     pts_rect = calib.lidar_to_rect(points[:, 0:3])
                #
                #     fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                #     pts_fov = points[fov_flag]
                #     corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                #     num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                #
                #     for k in range(num_objects):
                #         flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                #         num_points_in_gt[k] = flag.sum()
                #     annotations['num_points_in_gt'] = num_points_in_gt
                    # print("num_points_in_gt")
                    # print(num_points_in_gt)

            return info, seq_objects

        def process_single_seq(seq_idx, frame_idxs, delta_frame=1):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            infos = []
            # get label for each seq
            if has_label:
                seq_objects = self.get_label(int(seq_idx))
            else:
                seq_objects = None
            # get gps/imu pos
            seq_otxs = self.get_oxts_seq(seq_idx)
            # get calib for each seq
            seq_calib = self.get_calib(int(seq_idx))
            seq_file_path = self.root_split_path / 'velodyne' / ('%s'%seq_idx)

            frame_num = len(os.listdir(seq_file_path))
            # frame_idx_list = range(frame_num-delta_frame)
            # with futures.ThreadPoolExecutor(num_workers) as executor:
            #     infos = executor.map(process_single_scene(), [seq_idx, frame_idx_list, seq_objects, seq_calib])

            # 处理各个frame
            f_label = None
            f_lidar = None
            if eval_aug:
                f_label = open('/media/shenyl/sweeper1/kitti_tracking/aug_data/training/label_02/'+'%04d.txt'%int(seq_idx), 'w')
            for frame_idx in frame_idxs:
                frame_idx = int(frame_idx)
                d_frame = random.randint(1, delta_frame)
                if frame_idx + 1 > int(frame_idxs[-1]):
                    continue
                if ((int(seq_idx)==1) & (frame_idx == 176)):
                    d_frame = 5
                if ((int(seq_idx)==1) & ((frame_idx >176) & (frame_idx < 181))):
                    continue
                while ((frame_idx + d_frame > int(frame_idxs[-1])) | ((int(seq_idx) == 1)&(frame_idx + d_frame > 176) & (frame_idx + d_frame < 181))):
                    d_frame = random.randint(1, delta_frame)
                # if ((int(seq_idx)==1) & (frame_idx == 176)):
                #     info1 = process_single_scene(seq_idx, frame_idx, seq_objects, seq_calib, seq_otxs)
                #     info2 = process_single_scene(seq_idx, frame_idx + d_frame, seq_objects, seq_calib, seq_otxs)
                #     infos.append([info1, info2])
                # elif ((int(seq_idx)==1) & ((frame_idx >176) & (frame_idx < 181))):
                #     continue
                # else:

                lidar_file = '/media/shenyl/sweeper1/kitti_tracking/aug_data/training/velodyne/' + '%04d/'% int(seq_idx)
                if not os.path.exists(lidar_file):
                    os.makedirs(lidar_file)
                if frame_idx == int(frame_idxs[0]):
                    info1, seq_objects = process_single_scene(seq_idx, frame_idx, seq_objects, seq_calib, seq_otxs, f_label, lidar_file)
                else:
                    info1, seq_objects = process_single_scene(seq_idx, frame_idx, seq_objects, seq_calib, seq_otxs)
                if eval_aug:
                    info2, seq_objects = process_single_scene(seq_idx, frame_idx+d_frame, seq_objects, seq_calib, seq_otxs, f_label, lidar_file)
                else:
                    info2, seq_objects = process_single_scene(seq_idx, frame_idx + d_frame, seq_objects, seq_calib, seq_otxs)
                infos.append([info1, info2])
            if eval_aug:
                f_label.close()
            return infos

        infolist = []
        # 分seq处理,生成info
        for seq_idx in self.seq_id_list:
        # for seq_idx in range(21):
            # if seq_idx == seq_idx0:
            #     continue
            idx_file = self.split_dir / Path("%04d.txt"%seq_idx)
            frame_idxs = [x for x in open(idx_file).readlines()]
            print("idx_file\n", idx_file)
            print("frame idx\n", frame_idxs)
            seq_idx = "%04d"%seq_idx
            info = process_single_seq(seq_idx, frame_idxs, delta_frame)
            infolist = infolist + info
        return infolist

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k][0]
            seq_idx = info['idx_info']['seq_idx']
            sample_idx = info['idx_info']['frame_idx']
            points = self.get_lidar(seq_idx, sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%s_%d.bin' % (seq_idx, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None, last_pred_dicts=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if 'frame2_image_shape' in batch_dict:
                seq_idx = batch_dict['frame2_seq_id']
                calib = self.get_calib(seq_idx)
                # calib = batch_dict['frame2_calib'][batch_index]
                image_shape = batch_dict['frame2_image_shape'][batch_index]
            else:
                seq_idx = batch_dict['seq_id']
                calib = self.get_calib(seq_idx)
                # calib = batch_dict['calib'][batch_index]
                image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []

        for index, box_dict in enumerate(pred_dicts):
            if 'frame2_frame_id' in batch_dict:
                frame_id = batch_dict['frame2_frame_id'][index].item()
                seq_idx = batch_dict['frame2_seq_id'][index].item()
            else:
                frame_id = batch_dict['frame_id'][index].item()
                seq_idx = batch_dict['seq_id'][index].item()
            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%04d' % int(seq_idx))
                if not os.path.exists(cur_det_file):
                    os.makedirs(cur_det_file)
                with open(str(cur_det_file) + ('/%06d.txt' % int(frame_id)), 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        # print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                        #       % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                        #          bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                        #          dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                        #          loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                        #          single_pred_dict['score'][idx]), file=f)
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)
        return annos

    # @staticmethod
    # 已添加track_id prediction
    def generate_prediction_dicts_tracking(self, batch_dict_all, pred_dicts, class_names, output_path=None, trackers = None, last_pred_dicts=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'track_id': np.zeros(num_samples), 'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7]),
                'appear_life': np.zeros(num_samples), 'miss_time': np.zeros(num_samples)
            }
            return ret_dict

        def rotationMatrixToEulerAngles(R):
            # Checks if a matrix is a valid rotation matrix.
            def isRotationMatrix(R):
                Rt = np.transpose(R)
                shouldBeIdentity = np.dot(Rt, R)
                I = np.identity(3, dtype=R.dtype)
                n = np.linalg.norm(I - shouldBeIdentity)
                return n < 1e-6

            assert (isRotationMatrix(R))

            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

            singular = sy < 1e-6

            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])
                y = math.atan2(-R[2, 0], sy)
                z = math.atan2(R[1, 0], R[0, 0])
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0

            return np.array([x, y, z])

        # def generate_single_sample_dict(batch_index, box_dict, batch_dict1, trackers, min_appear_life=3):
        def generate_single_sample_dict(batch_index, box_dict, batch_dict1, trackers, min_appear_life=3, last_box_dict=None):
            global first_frame
            global det_none
            # 前后两帧点云利用svd计算相对位姿,并利用相对位姿得到box1在第二帧下的预测结果
            def box12_svd(box1, pc1_box, flow_box):
                # calculate R, t
                pc12_box = pc1_box + flow_box
                # viz_util.show_pcs(pc1_box, pc12)
                pc1 = np.transpose(pc1_box)
                pc12 = np.transpose(pc12_box)
                pc1_center = np.mean(pc1, axis=1, keepdims=True)
                pc12_center = np.mean(pc12, axis=1, keepdims=True)
                pc1_centered = pc1 - pc1_center
                pc12_centered = pc12 - pc12_center

                pc1_centered[2, :] = 0
                pc12_centered[2, :] = 0

                # 将pc12在局部坐标系下中心对称
                pc12_sym_centered = pc12_centered.copy()
                pc1_sym_centered = pc1_centered.copy()
                pc12_sym_centered = -pc12_sym_centered
                pc1_sym_centered = -pc1_sym_centered
                pc12_sym_centered = np.concatenate((pc12_centered, pc12_sym_centered), 1)
                pc1_sym_centered = np.concatenate((pc1_centered, pc1_sym_centered), 1)

                print("pc1\n", pc1_centered.shape)
                print("pc12\n", pc12_centered.shape)
                print("pc12_sym\n", pc12_sym_centered.shape)
                box1_centered = box1.copy()
                box1_centered[:3]=0
                # V.draw_three_scenes(points=np.transpose(pc1_sym_centered), points2=np.transpose(pc12_sym_centered), boxes1=box1.reshape(1,7))
                # mlab.show(stop=True)


                H = np.matmul(pc1_sym_centered, np.transpose(pc12_sym_centered))
                # H = np.matmul(pc1_centered, np.transpose(pc12_centered))

                U, S, Vt = np.linalg.svd(H)
                R = np.transpose(Vt) @ np.transpose(U)
                if np.linalg.det(R) < 0:
                    print("det(R)<R reflection detected, correcting for it ...")
                    Vt[2, :] *= -1
                    R = np.transpose(Vt) @ np.transpose(U)

                # print("R " + str(R))
                # print("pc12_center " + str(pc12_center))
                # print("pc1_center " + str(pc1_center))
                # t = -R @ pc1_center + pc12_center
                # print("t " + str(t))
                # change box1 to box12 with R t
                euler_angle = rotationMatrixToEulerAngles(R)
                print("euler angle: " + str(euler_angle))
                y_angle = euler_angle[2]
                # print(R)
                # print(z_angle)
                box12 = box1.copy()
                # print("t ", t.reshape(-1))
                # print(y_angle)
                t = np.mean(flow_box, axis=0, keepdims=True)
                box12[0:3] = box12[0:3] + t.reshape(-1)
                # box12[0, 0:3] = box12[0, 0:3] + np.transpose(t)
                box12[6] = box12[6] + y_angle



                # print(box12)
                # viz_util.show_pcs(pc1_box, pc12_box)
                # viz_util.show_two_3dboxes(box1, pc1_box, box12, pc12_box)
                return box12

            def association_with_Hungary(boxes12, boxes2, trackers=None, dets=None, max_miss_time=3):
                from scipy.optimize import linear_sum_assignment as linear_assignment
                if (boxes12.shape[0] > 0) & (boxes2.shape[0] > 0):
                    bev_ious = box_utils.boxes3d_nearest_bev_iou(boxes12, boxes2)
                    # print("bev ious\n", bev_ious)
                    indexs1, indexs2 = linear_assignment(-bev_ious)
                    # print("matches\n", indexs1, "\n", indexs2)
                if boxes2.shape[0] > 0:
                    unmatch_lists2 = np.array(range(boxes2.shape[0]))
                for index1 in range(len(trackers)):

                    if boxes2.shape[0] > 0:
                        if len(np.argwhere(indexs1==index1)) == 0:
                            # print(index1, " iou较小")
                            trackers[index1].miss_time = trackers[index1].miss_time + 1
                            # print("miss time\n", trackers[index1].miss_time)
                            trackers[index1].appear_life = 0
                        else:
                            # 该轨迹框没有找到匹配
                            i = (np.argwhere(indexs1==index1))[0,0]
                            index2 = indexs2[i]
                            # print(index1, " ", index2)
                            # print(bev_ious[index1][index2])
                            if bev_ious[index1][index2] > iou_threshold:
                                if trackers[index1].label != dets[index2].label:  #排除匹配框类别不同的情况
                                    continue
                                trackers[index1].miss_time = 0
                                trackers[index1].update(dets[index2]) # 用关联的当前帧的检测结果赋值给上一帧对应的tracker
                                unmatch_lists2[index2] = -1
                            # tracker没有找到det匹配,appear_life归零,miss_time加1
                            else:
                                trackers[index1].miss_time = trackers[index1].miss_time + 1
                                # print("miss time\n", trackers[index1].miss_time)
                                trackers[index1].appear_life = 0
                    else:
                        for index1 in range(len(trackers)):
                            trackers[index1].miss_time = trackers[index1].miss_time + 1
                            trackers[index1].appear_life = 0

                ##删除miss time大于阈值的轨迹框,从后往前删除
                tracker_len = len(trackers)
                for i in range(len(trackers)):
                    if trackers[tracker_len-i-1].miss_time > max_miss_time:
                        trackers.pop(tracker_len-i-1)

                # # 删除置信度小于阈值的轨迹框
                # tracker_len = len(trackers)
                # for i in range(len(trackers)):
                #     if trackers[tracker_len-i-1].score < DEL_TH:
                #         trackers.pop(tracker_len-i-1)


                #
                # print("tracks num ", len(trackers))
                # print("dets num ", len(dets))
                # print("dets boxes num ", boxes2.shape[0])
                # 对没有找到匹配的det,新建track并插入到tracker中
                if boxes2.shape[0] > 0:
                    for index2 in unmatch_lists2:
                        if index2 == -1:
                            continue
                        # print("添加轨迹框")
                        tracker.total_num = tracker.total_num + 1
                        t = tracker(dets[index2], track_id=tracker.total_num)
                        trackers.append(t)
                return trackers


            # def association(boxes12, boxes2, trackers=None, dets=None, max_miss_time=3):
            def association(boxes12, boxes2, trackers=None, dets=None, max_miss_time=3):
                from kitti_tracking.tracker import tracker, det
                bev_ious = box_utils.boxes3d_nearest_bev_iou(boxes12, boxes2)
                unmatch_lists2 = np.array(range(boxes2.shape[0]))
                print("box1 size", str(bev_ious.shape[0]), " box2 size: ", str(bev_ious.shape[1]))

                box1_num = bev_ious.shape[0]
                for i in range(box1_num):
                    index1 = box1_num-1-i  # 从后往前遍历,因为涉及到list删除操作
                    bev_iou = bev_ious[index1]
                    max_bev_iou = bev_iou.max()
                    # tracker找到了det匹配,用det更新tracker
                    if max_bev_iou > iou_threshold:
                        index2 = (np.argwhere(bev_iou==max_bev_iou)[0,0]).cpu().item()
                        # print("match: ", i, " ", index2.cpu().numpy())
                        if index2 >= boxes2.shape[0]:  # box2数量比box1小的情况
                            continue
                        if trackers[index1].label != dets[index2].label:  #排除匹配框类别不同的情况
                            continue
                        trackers[index1].update(dets[index2]) # 用关联的当前帧的检测结果赋值给上一帧对应的tracker,同时
                        unmatch_lists2[index2] = -1
                    # tracker没有找到det匹配,appear_life归零,miss_time加1
                    else:
                        trackers[index1].miss_time = trackers[index1].miss_time + 1
                        trackers[index1].appear_life = 0
                        # 对miss_time超出阈值的框进行删除操作
                        if trackers[index1].miss_time > max_miss_time:
                            trackers.pop(index1)

                print("tracks num ", len(trackers))
                # 对没有找到匹配的det,新建track并插入到tracker中
                for index2 in unmatch_lists2:
                    if index2 == -1:
                        continue
                    tracker.total_num= tracker.total_num+1
                    t = tracker(dets[index2], track_id = tracker.total_num)
                    trackers.append(t)

                return trackers

            from pcdet.datasets.kitti_tracking.tracker import tracker, det
            ### 确定是否是first_frame
            global last_seq_id

            seq_id = batch_dict1['seq_id'][0].cpu().numpy()
            frame_id = batch_dict1['frame_id'][0].cpu().numpy()
            print("seq: ", str(seq_id), " frame: ", str(frame_id))
            if not seq_id == last_seq_id:
                first_frame = True

            ###检测
            # 检测框
            det_scores = box_dict['pred_scores'].cpu().numpy()
            det_boxes = box_dict['pred_boxes'].cpu().numpy()
            det_labels = box_dict['pred_labels'].cpu().numpy()
            # 构建当前帧的dets
            dets = []
            for i in range(box_dict['pred_scores'].shape[0]):
                d = det(det_boxes[i], det_labels[i], det_scores[i])
                dets.append(d)
            print("det len ", len(dets))

            ### 跟踪预测值
            if first_frame | det_none: ## 第一帧,或者从初始帧到当前帧一直没有检测结果的情况下,直接将当前帧的检测结果都插入trackers中
                # det_scores_pre = box_dict['pred_scores'].cpu().numpy()
                # det_boxes_pre = box_dict['pred_boxes'].cpu().numpy()
                # det_labels_pre = box_dict['pred_labels'].cpu().numpy()
                det_scores_pre = last_box_dict['pred_scores'].cpu().numpy()
                det_boxes_pre = last_box_dict['pred_boxes'].cpu().numpy()
                det_labels_pre = last_box_dict['pred_labels'].cpu().numpy()
                if det_scores_pre.shape[0] == 0:
                    det_none = True
                    pred_dicts = get_template_prediction(0)
                    assert pred_dicts['bbox'].shape[0] == pred_dicts['track_id'].shape[0]
                    first_frame = False
                    return None, pred_dicts, []
                else:
                    det_none = False

                for i in range(det_scores_pre.shape[0]):
                    d = det(det_boxes_pre[i], det_labels_pre[i], det_scores_pre[i])
                    tracker.total_num = tracker.total_num+1
                    t=tracker(d,tracker.total_num)
                    trackers.append(t)
                import copy
                trackers_pre = copy.deepcopy(trackers)
                print("trackers_pre len ", len(trackers_pre))
            print("trackers len ", len(trackers))

            # center flow 场景流预测值
            pred_flow = box_dict['pred_center_flow'][0].permute(1, 0).cpu().numpy()
            # trackers的boxes
            trackers_boxes = np.array([tracker.boxes_lidar for tracker in trackers])
            # 对于第一帧的点,利用场景流移动到第二帧,利用SVD得到第一帧框到第二帧的变换,得到第二帧下预测框
            pred_centers_pre = last_box_dict['pred_centers'].cpu().numpy()  # 前一帧的所有center点
            boxes12 = np.zeros(trackers_boxes.shape, dtype=np.float32)
            if not len(trackers) == 0:
                corners_pre = box_utils.boxes_to_corners_3d(trackers_boxes)
                for index, box1 in enumerate(trackers_boxes):
                    # 截取框内的点和框内的场景流
                    inbox_flag_pre = box_utils.in_hull(pred_centers_pre, corners_pre[index])
                    if inbox_flag_pre.sum() == 0: # 对于内部点为0的框
                        box12 = box1
                    else:
                        pc1_inbox = pred_centers_pre[inbox_flag_pre].copy()
                        flow_inbox = pred_flow[inbox_flag_pre]
                        box12 = box12_svd(box1, pc1_inbox, flow_inbox)
                        trackers[index].predict(box12) # 利用场景流预测tracker在第二帧下的位置和尺寸
                    boxes12[index] = box12
                # 对第一帧检测框,第二帧检测框,第一帧基于场景流的预测框进行可视化
                if show:
                    # if (frame_id > 120) & (frame_id < 123):
                    # if (seq_id == 18) & (frame_id>160):
                    # if seq_id == 17:
                    if True:
                        print("白色点:第一帧点云; 红色点:第二帧点云; 绿色点:场景流估计的第二帧点云; \n 蓝色框:第一帧检测框; 绿色框:第二帧检测框; 红色框:第一帧基于场景流的预测框")
                        pred_centers = box_dict['pred_centers'].cpu().numpy()  # 前一帧的所有center点




                        V.draw_three_scenes(points=pred_centers_pre, points2=pred_centers,
                                            points3=pred_centers_pre + pred_flow, boxes1=trackers_boxes,
                                            boxes2=det_boxes, boxes3=boxes12)
                        mlab.show(stop=True)
                trackers_boxes = np.array([tracker.boxes_lidar for tracker in trackers])

            # trackers和box12对应,需要进行删除,添加的操作
            # 如果当前帧检测值为0,就不再进行关联,直接使用上一帧predict之后的trackers
            trackers = association_with_Hungary(boxes12, det_boxes, trackers, dets, max_miss_time=MAX_MISS_TIME)
            # trackers = association(boxes12, det_boxes, trackers, dets)


            # kitti_tracking形式的检测和跟踪结果,前一帧,因为当前帧联合前一帧决定前一帧的track_id
            calib = self.get_calib(seq_id)
                # batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            #如果是初始帧,那么也会返回前一帧的pred_dict
            pred_dict_pre = None
            if first_frame:
                # 将激光坐标系下的框转化到图像坐标系下
                det_boxes_camera_pre = box_utils.boxes3d_lidar_to_kitti_camera(det_boxes_pre, calib)
                det_boxes_img_pre = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    det_boxes_camera_pre, calib, image_shape=image_shape
                )
                pred_dict_pre = get_template_prediction(det_boxes_pre.shape[0])
                pred_dict_pre['name'] = np.array([class_names[tracker.label - 1] for tracker in trackers_pre])
                pred_dict_pre['alpha'] = -np.arctan2(-trackers_boxes[:, 1],
                                                 trackers_boxes[:, 0]) + det_boxes_camera_pre[:, 6]
                pred_dict_pre['bbox'] = det_boxes_img_pre
                pred_dict_pre['dimensions'] = det_boxes_camera_pre[:, 3:6]
                pred_dict_pre['location'] = det_boxes_camera_pre[:, 0:3]
                pred_dict_pre['rotation_y'] = det_boxes_camera_pre[:, 6]
                pred_dict_pre['score'] = np.array([tracker.score for tracker in trackers_pre])
                pred_dict_pre['boxes_lidar'] = trackers_boxes
                # 各个点场景流预测值
                pred_dict_pre['center_flows'] = pred_flow
                # 各个框track id预测值
                pred_dict_pre['track_id'] = np.array([tracker.track_id for tracker in trackers_pre])
                pred_dict_pre['frame_id'] = frame_id

                print("track pre len!!!\n", len(trackers_pre))
                print("bbox\n", pred_dict_pre['bbox'].shape[0])
                print("track id\n", pred_dict_pre['track_id'].shape[0])

            # 从trackers中选取appear_life大于阈值的框返回
            trackers_boxes = np.array([tracker.boxes_lidar for tracker in trackers if ((tracker.appear_life >= min_appear_life) | (frame_id<=min_appear_life))])
            # trackers_boxes = np.array([tracker.boxes_lidar for tracker in trackers if
            #                            (tracker.score >= OUTPUT_TH)])
            trackers_appear_life = np.array([tracker.appear_life for tracker in trackers])
            pred_dict = get_template_prediction(trackers_boxes.shape[0])
            if not (trackers_boxes.shape[0] == 0): # 如果某一帧的tracker数量为0
                trackers_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(trackers_boxes, calib)
                trackers_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                    trackers_boxes_camera, calib, image_shape=image_shape
                )
                # pred_dict = get_template_prediction(trackers_boxes.shape[0])
                pred_dict['name'] = np.array([class_names[tracker.label - 1] for tracker in trackers if ((tracker.appear_life >= min_appear_life) | (frame_id<=min_appear_life))])
                # pred_dict['name'] = np.array([class_names[tracker.label - 1] for tracker in trackers if (tracker.score >= OUTPUT_TH)])

                pred_dict['alpha'] = -np.arctan2(-trackers_boxes[:, 1],
                                                 trackers_boxes[:, 0]) + trackers_boxes_camera[:, 6]
                pred_dict['bbox'] = trackers_boxes_img
                pred_dict['dimensions'] = trackers_boxes_camera[:, 3:6]
                pred_dict['location'] = trackers_boxes_camera[:, 0:3]
                pred_dict['rotation_y'] = trackers_boxes_camera[:, 6]
                pred_dict['score'] = np.array([tracker.score for tracker in trackers if ((tracker.appear_life >= min_appear_life) | (frame_id<=min_appear_life))])
                # pred_dict['score'] = np.array([tracker.score for tracker in trackers if
                #             (tracker.score >= OUTPUT_TH)])
                pred_dict['boxes_lidar'] = trackers_boxes
                # 各个点场景流预测值
                pred_dict['center_flows'] = pred_flow
                # 各个框track id预测值
                pred_dict['track_id'] = np.array([tracker.track_id for tracker in trackers if ((tracker.appear_life >= min_appear_life) | (frame_id<=min_appear_life))])
                # pred_dict['track_id'] = np.array([tracker.track_id for tracker in trackers if (tracker.score >= OUTPUT_TH)])

                pred_dict['frame_id'] = frame_id
                assert pred_dict['track_id'].shape[0] == trackers_boxes_camera.shape[0] == trackers_boxes.shape[0] == pred_dict['name'].shape[0] == pred_dict['score'].shape[0]
                print("trackers num: ", pred_dict['track_id'].shape[0])
            if first_frame == True:
                first_frame = False
            last_seq_id = seq_id
            return pred_dict_pre, pred_dict, trackers

        batch_dict = {}
        batch_dict2 = {}
        global last_seq_idx
        global annos_file
        for key, val in batch_dict_all.items():
            if key.split("_", 1)[0] == "frame1":
                batch_dict[key.split("_", 1)[1]] = val
                continue
            if key.split("_", 1)[0] == "frame2":
                batch_dict2[key.split("_", 1)[1]] = val
                continue
            batch_dict[key] = val
            batch_dict2[key] = val

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_idx = batch_dict['frame_id'][index].cpu().numpy()
            seq_idx = batch_dict['seq_id'][index].cpu().numpy()
            single_pred_dict1, single_pred_dict2, trackers = generate_single_sample_dict(index, box_dict, batch_dict, trackers, last_box_dict=last_pred_dicts[0], min_appear_life=MIN_APPEAR_LIFE)
            # single_pred_dict = generate_single_sample_dict(index, box_dict)
            tracker_box = np.array([t.boxes_lidar for t in trackers])

            if not single_pred_dict1 == None:
                single_pred_dict1['seq_id'] = seq_idx
                single_pred_dict1['frame_id'] = frame_idx
                # annos.append(single_pred_dict1)
                annos_file.append(single_pred_dict1)

            if not single_pred_dict2 == None:
                single_pred_dict2['seq_id'] = seq_idx
                single_pred_dict2['frame_id'] = frame_idx + 1
                annos.append(single_pred_dict2)
                annos_file.append(single_pred_dict2)
            else:
                single_pred_dict2 = {}
                single_pred_dict2['seq_id'] = seq_idx
                single_pred_dict2['frame_id'] = frame_idx + 1
                annos.append(single_pred_dict2)
                annos_file.append(single_pred_dict2)


            if output_path is not None:
                if frame_idx == end_frame_lists[str(self.dataset_cfg.DATA_SPLIT['test'])][int(seq_idx)]:
                    cur_det_path = output_path / 'det_results/data'
                    cur_track_path = output_path / 'track_results/data'
                    if not os.path.exists(cur_det_path):
                        os.makedirs(cur_det_path)
                    if not os.path.exists(cur_track_path):
                        os.makedirs(cur_track_path)
                    cur_det_file = cur_det_path / ('%04d.txt' % last_seq_idx)
                    cur_track_file = cur_track_path / ('%04d.txt' % last_seq_idx)
                    print("cur_det_file: ", str(cur_det_file))
                    with open(cur_det_file, 'w') as f2:
                        with open(cur_track_file, 'w') as f1:
                            for i, a in enumerate(annos_file):
                                bbox = a['bbox']
                                loc = a['location']
                                dims = a['dimensions']  # lhw -> hwl

                                if len(bbox) == 0:
                                    continue
                                print("bbox len\n", len(bbox))
                                print("track id len\n",len(a['track_id']))
                                for idx in range(len(bbox)):
                                    if not a['track_id'][idx] == -1:
                                        idx = len(bbox) - 1 - idx
                                        # 输出多目标跟踪结果
                                        print('%d %d %s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                                              % (a['frame_id'], a['track_id'][idx], a['name'][idx], a['alpha'][idx],
                                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                                 loc[idx][1], loc[idx][2], a['rotation_y'][idx],
                                                 a['score'][idx]), file=f1)

                                        # 输出AB3DMOT要求的检测结果
                                        print('%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f'
                                            % (a['frame_id'], NAME2ID[a['name'][idx]],
                                               bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                               a['score'][idx],
                                               dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                               loc[idx][1], loc[idx][2], a['rotation_y'][idx],
                                               a['alpha'][idx]), file=f2)


                    annos_file = []
            last_seq_idx = seq_idx
        return annos, trackers

    # 目前只对前一帧的检测结果进行评估, todo:kitti_tracking evaluation
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0][0].keys():
            return None, {}

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info[1]['annos']) for info in self.kitti_infos]
        for i, g in enumerate(eval_gt_annos):
            if g == None:
                print("None -5", str(i))
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # 场景流评估
        return ap_result_str, ap_dict

def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4, eval_aug=False):
    dataset = KittiTracking(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'
    if eval_aug:
        val_filename = save_path / ('kitti_infos_%s_aug.pkl' % val_split)
    else:
        train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
        val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
        trainval_filename = save_path / 'kitti_infos_trainval.pkl'
        test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')
    if eval_aug == False:
        dataset.set_split(train_split)
        kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, eval_aug=eval_aug)
        with open(train_filename, 'wb') as f:
            pickle.dump(kitti_infos_train, f)
        print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True, eval_aug=eval_aug)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    if eval_aug == False:
        with open(trainval_filename, 'wb') as f:
            pickle.dump(kitti_infos_train + kitti_infos_val, f)
        print('Kitti info trainval file is saved to %s' % trainval_filename)

        dataset.set_split('test')
        kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
        with open(test_filename, 'wb') as f:
            pickle.dump(kitti_infos_test, f)
        print('Kitti info test file is saved to %s' % test_filename)

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    # 生成info
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        eval_aug = False ###True:在生成eval info的时候人为添加大运动
        # ROOT_DIR = Path("/media/shenyl/sweeper1/kitti_tracking/")
        ROOT_DIR = Path(dataset_cfg.DATA_PATH)
        if eval_aug:
            ROOT_DIR = Path("/media/shenyl/sweeper1/kitti_tracking/aug_data/")
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR,
            save_path=ROOT_DIR,
            eval_aug=eval_aug,
        )

    else:
        #载入配置文件
        # yaml_path = "/home/shenyl/Documents/code/SSDT/tools/cfgs/dataset_configs/kitti_tracking_dataset.yaml"
        yaml_path = "/home/shenyl/Documents/code/SSDT/tools/cfgs/dataset_configs/kitti_tracking_dataset_voxel.yaml"
        f_yaml = open(yaml_path, encoding="UTF-8")
        cfg = EasyDict(yaml.load(f_yaml, Loader=yaml.FullLoader))
        # 设置root_path
        # root_path = Path("/media/shenyl/sweeper1/kitti_tracking/")
        root_path = Path(cfg.DATA_PATH)
        # root_path = Path("/media/shenyl/sweeper1/kitti_tracking/aug_data")
        train_dataset = KittiTracking(cfg, ['Car', 'Pedestrian', 'Cyclist'], training=True, root_path=root_path, logger=None)
        eval_dataset = KittiTracking(cfg, ['Car', 'Pedestrian', 'Cyclist'], training=False, root_path=root_path, logger=None)
        # print(list(DataLoader(train_dataset, batch_size=2, pin_memory = True, num_workers=1,
        # shuffle = True, collate_fn = train_dataset.collate_batch,
        # drop_last = False, timeout = 0)))
        print(list(DataLoader(train_dataset, batch_size=1, shuffle=False)))


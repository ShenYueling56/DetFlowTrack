from functools import partial

import numpy as np

from ...utils import common_utils, box_utils
from . import augmentor_utils, database_sampler

# import mayavi.mlab as mlab
# from tools.visual_utils import visualize_utils as V


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        print("gt sampling")
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None, aug_dict=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        enable=-1
        if aug_dict is not None:
            enable = aug_dict['flip_enable']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable= getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points, enable
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if aug_dict is not None:
            aug_dict['flip_enable'] = enable
        return data_dict, aug_dict

    def random_world_rotation(self, data_dict=None, config=None, aug_dict=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        world_rot_enable = -1
        world_rot_angle = 0
        if aug_dict is not None:
            world_rot_enable = aug_dict['world_rot_enable']
            world_rot_angle = aug_dict['world_rot_angle']
        gt_boxes, points, enable, rot_angle = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range, enable=world_rot_enable, rot_angle=world_rot_angle
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if aug_dict is not None:
            aug_dict['world_rot_enable'] = enable
            aug_dict['world_rot_angle'] = rot_angle
        return data_dict, aug_dict

    def random_world_scaling(self, data_dict=None, config=None, aug_dict=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        world_scale_enable=-1
        world_scale=1
        if aug_dict is not None:
            world_scale_enable = aug_dict['world_scale_enable']
            world_scale = aug_dict['world_scale']
        gt_boxes, points, enable, scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE'], world_scale_enable, world_scale
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        if aug_dict is not None:
            aug_dict['world_scale_enable'] = enable
            aug_dict['world_scale'] = scale
        return data_dict, aug_dict

    def boxes_dis(self, data_dict=None, config=None, aug_dict=None):
        import copy
        if data_dict is None:
            return partial(self.boxes_dis, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes.reshape(-1, 7)) # 各个框的顶点
        # 可视化变换之前
        # V.draw_scenes(points=points, gt_boxes=gt_boxes)
        # mlab.show(stop=True)
        #对各个框分别操作
        for i, box3d in enumerate(gt_boxes):
            p = np.random.rand()
            if p > config['DIS_AUG_PROB']:
                # 随机获取平移量
                displacement = np.random.randn(3) * config['MAX_DIS']
                displacement[2] = 0

                # 判断平移后的框是否会和其余的框重合,如果是,就不进行此框的平移
                new_box3d = copy.deepcopy(gt_boxes[i])
                new_box3d[0:3] = new_box3d[0:3] + displacement
                new_enlarged_box3d = copy.deepcopy(new_box3d)
                new_enlarged_box3d[3] += 0.5
                new_enlarged_box3d[4] += 0.5
                new_enlarged_box3d[5] += 2
                if i > 1:
                    iou = box_utils.boxes3d_nearest_bev_iou(new_enlarged_box3d.reshape(1, 7), gt_boxes[:i, :])
                    if iou.max() > 1e-8:
                        continue
                if i+1 < gt_boxes.shape[0]:
                    iou = box_utils.boxes3d_nearest_bev_iou(new_enlarged_box3d.reshape(1, 7), gt_boxes[i+1:, :])
                    if iou.max() > 1e-8:
                        continue

                # 将平移量作用于框内的点
                pt_mask_flag = box_utils.in_hull(points[:, :3], corners_lidar[i])
                points[pt_mask_flag, :3] = points[pt_mask_flag, :3] + displacement
                # 将平移量作用于框:
                gt_boxes[i, 0:3] = gt_boxes[i, 0:3] + displacement

                # 去掉原本点云中在平移后的框范围内的点
                enlarged_corners_lidar = box_utils.boxes_to_corners_3d(
                    new_enlarged_box3d.reshape(-1, 7))  # 各个框的顶点
                pt_del_mask_flag = box_utils.in_hull(points[:, :3], enlarged_corners_lidar[0])
                points = points[(pt_del_mask_flag < 1) | (pt_mask_flag)]


        # 可视化变换之后
        # V.draw_scenes(points=points, gt_boxes=gt_boxes)
        # mlab.show(stop=True)

        data_dict['points'] = points
        data_dict['gt_boxes'] = gt_boxes
        return data_dict, aug_dict

    def boxes_rot(self, data_dict=None, config=None, aug_dict=None):
        import copy
        if data_dict is None:
            return partial(self.boxes_rot, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes.reshape(-1, 7))# 各个框的顶点
        # 可视化变换之后
        # V.draw_scenes(points=points, gt_boxes=gt_boxes)
        # mlab.show(stop=True)

        for i, box3d in enumerate(gt_boxes):
            p = np.random.rand()
            if p > config['ROT_AUG_PROB']:
                # 随机获取绕z轴旋转角
                rot_angle = np.random.randn(1) * np.pi/(config['MAX_ANGLE'])

                # 将旋转角作用于框:
                new_gt_box = gt_boxes[i]
                new_gt_box[6] += rot_angle
                gt_boxes[i] = new_gt_box

                # 将旋转量作用于框内的点
                # 将点转换到局部坐标系下
                pt_mask_flag = box_utils.in_hull(points[:, :3], corners_lidar[i])
                new_points = points[pt_mask_flag, :3]- new_gt_box[:3]
                new_points = common_utils.rotate_points_along_z(new_points[np.newaxis, :, :3], np.array(rot_angle))[0]
                points[pt_mask_flag, :3] = new_points + new_gt_box[:3]

        # 可视化变换之后
        # V.draw_scenes(points=points, gt_boxes=gt_boxes)
        # mlab.show(stop=True)
        data_dict['points'] = points
        data_dict['gt_boxes'] = gt_boxes
        return data_dict, aug_dict

    def forward(self, data_dict, aug_dict=None, if_aug=True):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_augmentor in self.data_augmentor_queue:
            data_dict, aug_dict = cur_augmentor(data_dict=data_dict, aug_dict=aug_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'track_id' in data_dict:
                data_dict['track_id'] = data_dict['track_id'][gt_boxes_mask]
            data_dict.pop('gt_boxes_mask')
        return data_dict, aug_dict

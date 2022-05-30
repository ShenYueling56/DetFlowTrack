import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

import sys
sys.path.append("/home/shenyl/Documents/code/3DSSD-pytorch-openPCDet")
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import box_utils, common_utils, object3d_kitti, calibration_kitti, tracking_kitti
from visual_utils import visualize_utils as V


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', seq_index=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        if seq_index == None:
            data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
            print(data_file_list)
            self.seq_index = None
        else:
            self.seq_index = seq_index
            data_file_list = glob.glob(str(root_path / Path("%04d"%seq_index) / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(str(self.sample_file_list[index]), dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }
        # get calib file
        if self.seq_index == None:
            calib_file = Path('/media/shenyl/Elements/KITTI_object/object/training') / 'calib' / ('%06d.txt' % index)
        else:
            calib_file = Path('/media/shenyl/sweeper1/kitti_tracking/training') / 'calib' / Path("%04d"%self.seq_index + ".txt")
            print(calib_file)
        assert calib_file.exists()
        calib = calibration_kitti.Calibration(calib_file)
        # get label file
        if self.seq_index == None:
            label_file = Path('/media/shenyl/Elements/KITTI_object/object/training') / 'label_2' / ('%06d.txt' % index)
            assert label_file.exists()
            gt_object_list = object3d_kitti.get_objects_from_label(label_file)
            loc = np.concatenate([obj.loc.reshape(1, 3) for obj in gt_object_list if obj.cls_type != 'DontCare'], axis=0)
            dims = np.array([[obj.l, obj.h, obj.w] for obj in gt_object_list if obj.cls_type != 'DontCare'])
            rots = np.array([obj.ry for obj in gt_object_list if obj.cls_type != 'DontCare'])
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            input_dict.update({'gt_boxes_lidar': gt_boxes_lidar})
        else:
            label_file = Path('/media/shenyl/sweeper1/kitti_tracking/training') / 'label_02' / ('%04d.txt' % self.seq_index)
            assert label_file.exists()
            gt_object_list = tracking_kitti.get_objects_from_label(label_file)[index]
            loc = np.concatenate([obj.loc.reshape(1, 3) for obj in gt_object_list if obj.cls_type != 'DontCare'],
                                 axis=0)
            dims = np.array([[obj.l, obj.h, obj.w] for obj in gt_object_list if obj.cls_type != 'DontCare'])
            rots = np.array([obj.ry for obj in gt_object_list if obj.cls_type != 'DontCare'])
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)
            input_dict.update({'gt_boxes_lidar': gt_boxes_lidar})

        data_dict, _ = self.prepare_data(data_dict=input_dict)
        return data_dict



def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/3DSSD_openPCDet.yaml',
    #                     help='specify the config for demo')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/3DSSD_Multiscale_Det.yaml',
    #                     help='specify the config for demo')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/3DSSD_DetTrack_openPCDet.yaml',
    #                     help='specify the config for demo')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/SOLO_3DSSD_DetTrack_openPCDet.yaml',
                        help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='/media/shenyl/Elements/KITTI_object/object/training/velodyne/',
    #                     help='specify the point cloud data file or directory')
    parser.add_argument('--data_path', type=str,
                        default='/media/shenyl/sweeper1/kitti_tracking/training/velodyne/',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/shenyl/9999_result/0529/3DSSD/ckpt/20210529-140739/checkpoint_epoch_100.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, seq_index=0
    )


    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            gt_boxes = data_dict['gt_boxes_lidar'][0]
            pred_flow = pred_dicts[0]['pred_center_flow'][0].permute(1, 0)
            # print("pred flow shape: " + str(pred_flow.shape))
            centers1 = pred_dicts[0]['pred_centers_pre']
            centers2 = pred_dicts[0]['pred_centers']
            # print("center shape: " + str(centers1.shape))
            # gt_boxes = None
            # print("pred box num in frame1:" + str(pred_dicts[0]['pred_boxes_pre'].shape))
            # print("pred box num in frame2:"+str(pred_dicts[0]['pred_boxes'].shape))
            # 可视化第二帧检测结果
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # 可视化场景流结果
            V.draw_three_scenes(centers1, points2=centers2, points3=centers1, boxes1=pred_dicts[0]['pred_boxes_pre'], boxes2=pred_dicts[0]['pred_boxes'])
            mlab.show(stop=True)

    logger.info('Demo done.')


if __name__ == '__main__':
    main()

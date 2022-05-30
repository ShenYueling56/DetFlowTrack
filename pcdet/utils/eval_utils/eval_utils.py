import pickle
import time

import numpy as np
import torch
import tqdm

import sys
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, tracking_calibration_kitti
from pcdet.datasets.kitti_tracking.tracker import tracker, det
import mayavi.mlab as mlab
from pcdet.utils.visual_utils import visualize_utils as V

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])




def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None, inference=False):
    global trackers
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)
        final_output_dir = result_dir / 'result'
        # 如果仅获得AB3DMOT需要的检测结果,则保存在如下地址
        # final_output_dir = result_dir / 'final_det_result' / 'data'
        if save_to_file:
            final_output_dir.mkdir(parents=True, exist_ok=True)
    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    for i, batch_dict in enumerate(dataloader):  # 以batch_size=1遍历全部数据集,由于shffle=false,所以是逐帧按顺序遍历的
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, pre_pred_dicts, ret_dict = model(batch_dict)
            if not pre_pred_dicts == []:
                last_pred_dicts = pre_pred_dicts

        disp_dict = {}

        # if 'pred_center_flow' in pred_dicts[0]:
            # 可视化场景流和检测结果
            # pred_flow = pred_dicts[0]['pred_center_flow'][0].permute(1, 0)
            # print("pred flow shape " + str(pred_flow.shape))
            # centers1 = pred_dicts[0]['pred_centers_pre']
            # centers2 = pred_dicts[0]['pred_centers']
            # print("center shape: " + str(centers1.shape))
            # gt_boxes = None
            # print("pred box num in frame1:" + str(pred_dicts[0]['pred_boxes_pre'].shape))
            # print("pred box num in frame2:" + str(pred_dicts[0]['pred_boxes'].shape))
            # print("box gt num in frame2:" + str(batch_dict['frame2_gt_boxes'][0].shape))
            # print("batch dict key: " + str(batch_dict.keys()))
            # 可视化第二帧检测结果
            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # 可视化场景流结果
            # print("白色点:第一帧点云; 红色点:第二帧点云; 绿色点:场景流估计的第二帧点云; \n 蓝色框:第一帧检测框; 绿色框:第二帧检测框; 红色框:第二帧真实值框")
            # V.draw_three_scenes(centers1, points2=centers2, points3=centers1+pred_flow, boxes1=pred_dicts[0]['pred_boxes_pre'],
            #                     boxes2=pred_dicts[0]['pred_boxes'], boxes3=batch_dict['frame2_gt_boxes'][0][:, :7])
            # mlab.show(stop=True)
            # 激光坐标系
            # print("白色点:第一帧点云; 红色点:第二帧点云;\n 蓝色框:第一帧检测框; 绿色框:第二帧检测框; 红色框:第二帧真实值框")
            # points1 = batch_dict['frame1_points'][:, 1:4].cpu().numpy()
            # points2 = batch_dict['frame2_points'][:, 1:4].cpu().numpy()
            # # V.draw_three_scenes(points1, points2=points2,
            # #                     # boxes1=pred_dicts[0]['pred_boxes_pre'],
            # #                     boxes2=pred_dicts[0]['pred_boxes'], boxes3=batch_dict['frame2_gt_boxes'][0][:, :7])
            # print(pred_dicts[0].keys())
            # boxes1 = torch.cat((pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_scores'].reshape(-1,1)), 1)
            # V.draw_three_scenes(points=points2,points2=pred_dicts[0]['pred_centers'],
            #                     boxes1=boxes1, boxes2=batch_dict['frame2_gt_boxes'][0][:, :7])
            #
            # mlab.show(stop=True)
        # boxes1 = torch.cat((last_pred_dicts[0]['pred_boxes'], last_pred_dicts[0]['pred_scores'].reshape(-1, 1)), 1)
        seq_idx = batch_dict['seq_id'] if 'seq_id' in batch_dict else batch_dict['frame1_seq_id']
        # if seq_idx == 18:
        #     boxes2 = torch.cat((pred_dicts[0]['pred_boxes'], pred_dicts[0]['pred_scores'].reshape(-1, 1)), 1)
        #     # V.draw_three_scenes(points=batch_dict['frame1_points'][:, 1:4].cpu().numpy(),
        #     #                     boxes1=boxes1, boxes2=batch_dict['frame1_gt_boxes'][0][:, :7])
        #     V.draw_three_scenes(points=batch_dict['frame2_points'][:, 1:4].cpu().numpy(),
        #                         boxes1=boxes2, boxes2=batch_dict['frame2_gt_boxes'][0][:, :7])
        #
        #     mlab.show(stop=True)

        statistics_info(cfg, ret_dict, metric, disp_dict)
        if cfg.DATA_CONFIG.DATASET == 'KittiTracking':
        # if False:

            frame_idx = batch_dict['frame_id'] if 'frame_id' in batch_dict else batch_dict['frame1_frame_id']
            if i == 0:
                # last_pred_dicts = pred_dicts
                last_seq_id = seq_idx
                trackers = []
                tracker.total_num = 0
            elif not seq_idx == last_seq_id:
                # last_pred_dicts = pred_dicts
                last_seq_id = seq_idx
                trackers = []
                tracker.total_num = 0
            if inference == True:
                # 每个seq写入一个txt
                annos, trackers = dataset.generate_prediction_dicts_tracking(
                    batch_dict, pred_dicts, class_names,
                    output_path=final_output_dir if save_to_file else None,
                    trackers=trackers, last_pred_dicts=last_pred_dicts
                )
            else:
                # 每个frame都写入一个txt
                annos = dataset.generate_prediction_dicts(
                    batch_dict, pred_dicts, class_names,
                    output_path=final_output_dir if save_to_file else None, last_pred_dicts=last_pred_dicts
                )
            last_pred_dicts = pred_dicts
        else:
            annos = dataset.generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=final_output_dir if save_to_file else None)
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        if not anno == None:
            total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict

def eval_single_ckpt(cfg, model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )

if __name__ == '__main__':
    pass

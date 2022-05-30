# coding=utf-8

import argparse
import datetime
import glob
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from pcdet.utils.train_utils.optimization import build_optimizer, build_scheduler
from pcdet.utils.train_utils.train_utils import train_model, eval_model
from eval import repeat_eval_ckpt

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/SOLO_3DSSD_DetTrack_openPCDet.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/3DSSD_openPCDet.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/3DSSD_Multiscale_Det.yaml', help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/fuse_3DSSD_DetTrack_openPCDet.yaml',
    #                     help='specify the config for training')
    # parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/3DSSD_openPCDet.yaml',
    #                                         help='specify the config for training')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn_DetTrack.yaml',
                                            help='specify the config for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='3DSSD', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default="../pretrained_weight/pv_rcnn_8369.pth", help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )

        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt' / Path('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    total_gpus = len(gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')/(Path('%s' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    # 构建数据集
    gpu_list = gpu_list.split(',')
    print("gpu list" + str(len(gpu_list)))
    total_gpus = len(gpu_list)

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size * total_gpus,
        dist=dist_train, workers=args.workers*args.batch_size * total_gpus,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,  # store true
        total_epochs=args.epochs,
        total_gpus=total_gpus,
    )

    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        root_path=Path(cfg.DATA_CONFIG.DATA_PATH),
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=dist_train, workers=args.workers, logger=logger, training=False,
        total_gpus=1,
    )

    # 构建网络
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # # 固定SSD det参数
    # for name, param in model.named_parameters():
    #     if ((name.split('.')[0]=='backbone_3d') | (name.split('.')[0] == 'point_head')):
    #         if ((name.split('.')[3]).split('_')[0] == 'flow'):
    #             continue
    #         print(name)
    #         param.requires_grad = False

    # 固定flow参数
    # for name, param in model.named_parameters():
    #     if (name.split('.')[0] == 'backbone_3d'):
    #         if ((name.split('.')[3]).split('_')[0] == 'flow'):
    #             print(name)
    #             param.requires_grad = False
    #     if (name.split('.')[0] == 'sceneflow_head'):
    #         print(name)
    #         param.requires_grad = False

    # 固定pvrcnn rpn阶段参数
    for name, param in model.named_parameters():
        if (name.split('.')[0] in ['backbone_3d', 'backbone_2d', 'dense_head']):
            print(name)
            param.requires_grad = False

    # 添加多gpu训练
    if len(gpu_list) > 1:
        print("Let's use", len(gpu_list), "GPUs!")
        # model = nn.DataParallel(model, device_ids=[0,1])
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        model.cuda()
    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # 载入预训练模型
    start_epoch = it = 0
    last_epoch = -1
    if args.pretrained_model is not None:
        pretrained_dict = torch.load(open(args.pretrained_model, 'rb'))['model_state']
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k.split('.')[0] == 'module':
                k = k[7:]
                if k.split('.')[0] in ['backbone_3d', 'backbone_2d', 'dense_head']:
                    new_pretrained_dict[k] = v
            else:
                if k.split('.')[0] in ['backbone_3d', 'backbone_2d', 'dense_head']:
                    new_pretrained_dict[k] = v

        if isinstance(model, nn.DataParallel):
            model_dict = model.module.state_dict()
            model_dict.update(new_pretrained_dict)
            model.module.load_state_dict(model_dict)
            # model.module.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)
        else:
            model_dict = model.state_dict()
            model_dict.update(new_pretrained_dict)
            model.load_state_dict(model_dict)
            # model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist, logger=logger)

    # 断点重新训练
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    # 载入原本文件夹中的模型继续训练
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            if len(gpu_list) > 1:
                it, start_epoch = model.module.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
                )
            else:
                it, start_epoch = model.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
                )
            last_epoch = start_epoch + 1

    model.train()

    if dist_train:
        #model = nn.parallel.DistributedDataParallel(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], output_device = [cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # 打印需要更新的参数名字和状态
    # print("requires grad")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    train_model(
        model,
        optimizer,
        train_loader,
        test_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        logger=logger
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    args.start_epoch = max(args.epochs, 0)  # Only evaluate the last epochs


    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, args, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()

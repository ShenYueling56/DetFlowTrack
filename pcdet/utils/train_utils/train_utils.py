import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from pcdet.config import cfg
from pcdet.models import load_data_to_gpu


def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()
        loss, tb_dict = model_func(model, batch)
        # print("loss ", loss.item())
        loss.backward() # loss=0的情况也需要backward(),因为backward()包含了清空buffer,防止显存溢出
        if not torch.isnan(loss):
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP) # 梯度剪裁,用于防止梯度消失和梯度爆炸
            optimizer.step()
            # disp_dict = {'loss': loss.item(), 'flow loss': tb_dict['total_flow_loss'].item(), 'lr': cur_lr}
        accumulated_iter += 1
        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            # tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def eval_one_epoch(model, eval_loader, accumulated_iter, rank, tbar, tb_log=None, leave_pbar=False):
    dataset = eval_loader.dataset
    class_names = dataset.class_names
    det_annos = []

    total_it_each_epoch =len(eval_loader)
    dataloader_iter = iter(eval_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='eval', dynamic_ncols=True)

    model.eval()
    for cur_it in range(total_it_each_epoch):
        try:
            batch_dict = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(eval_loader)
            batch_dict = next(dataloader_iter)
            print('new iters')

        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict = model(batch_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names
        )
        det_annos += annos
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.refresh()

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
    )


    if tb_log is not None:
        for j, cls in enumerate(class_names):
            tb_log.add_scalar('%s_3d/easy_R40' % cls, result_dict['%s_3d/easy_R40' % cls], accumulated_iter)
            tb_log.add_scalar('%s_3d/moderate_R40' % cls, result_dict['%s_3d/moderate_R40' % cls], accumulated_iter)
            tb_log.add_scalar('%s_3d/hard_R40' % cls, result_dict['%s_3d/hard_R40' % cls], accumulated_iter)
            # sceneflow eval loss
            # tb_log.add_scalar('sceneflow loss', ret_dict['sceneflow_loss'], accumulated_iter)
    accumulated_iter = accumulated_iter + 1
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader, eval_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, logger, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    eval_accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        train_total_it_each_epoch = len(train_loader)
        test_total_it_each_epoch = len(eval_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            train_total_it_each_epoch = len(train_loader) // max(total_epochs, 1)
            test_total_it_each_epoch = len(eval_loader) // max(total_epochs, 1)
        train_dataloader_iter = iter(train_loader)
        test_dataloader_iter = iter(eval_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler
            accumulated_iter = train_one_epoch(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=train_total_it_each_epoch,
                dataloader_iter=train_dataloader_iter
            )

            # 对模型进行精度评估
            if cur_epoch%5==0: #每5个epoch eval一次
                eval_accumulated_iter = eval_one_epoch(
                    model, eval_loader,
                    accumulated_iter=eval_accumulated_iter,
                    rank=rank, tbar=tbar, tb_log=tb_log,
                    leave_pbar=(cur_epoch + 1 == total_epochs),
                )

            # record this epoch which has been evaluated
            # logger.info('Epoch %s has been evaluated' % cur_epoch)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                # 删除多余的ckpt文件
                # if ckpt_list.__len__() >= max_ckpt_save_num:
                #     for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                #         os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def eval_model(model, eval_loader, model_func,
                start_epoch, total_epochs, start_iter, rank, tb_log, eval_sampler=None,
                merge_all_iters_to_one_epoch=False):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(eval_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(eval_loader.dataset, 'merge_all_iters_to_one_epoch')
            eval_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(eval_loader) // max(total_epochs, 1)

        dataloader_iter = iter(eval_loader)
        for cur_epoch in tbar:
            if eval_sampler is not None:
                eval_sampler.set_epoch(cur_epoch)

            # eval one epoch
            accumulated_iter = eval_one_epoch(
                model, eval_loader,
                accumulated_iter=accumulated_iter,
                rank=rank, tbar=tbar, tb_log=tb_log,
            )

def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)

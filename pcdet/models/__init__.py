from collections import namedtuple

import numpy as np
import torch

from .detectors import build_detector


def build_network(model_cfg, num_class, dataset):
    model = build_detector(
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape', 'frame1_calib', 'frame2_calib', 'frame1_image_shape', 'frame2_image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict = model(batch_dict)
        # print("ret dict loss ", ret_dict['loss'].item())
        # loss = ret_dict['loss'][ret_dict['loss']>0]
        loss = ret_dict['loss']
        loss = loss.mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict)

    return model_func

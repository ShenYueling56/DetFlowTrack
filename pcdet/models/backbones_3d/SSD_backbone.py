import torch
import torch.nn as nn

from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack

import torchsnooper

class SSDBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        self.channel_in = input_channels - 3
        self.channel_out_list = [self.channel_in]

        self.num_points_each_layer = []
        self.skip_channel_list = [input_channels - 3]

        self.sa_config = self.model_cfg.SA_CONFIG
        self.layer_types = self.model_cfg.SA_CONFIG.LAYER_TYPE
        self.ctr_indexes = self.model_cfg.SA_CONFIG.CTR_INDEX
        self.layer_names = self.model_cfg.SA_CONFIG.LAYER_NAME
        self.layer_inputs = self.model_cfg.SA_CONFIG.LAYER_INPUT
        self.max_translate_range = self.model_cfg.SA_CONFIG.MAX_TRANSLATE_RANGE

        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            self.channel_in = self.channel_out_list[self.layer_inputs[k]]
            if self.layer_types[k] == 'SA_Layer':
                self.mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
                for idx in range(self.mlps.__len__()):
                    self.mlps[idx] = [self.channel_in] + self.mlps[idx]

                self.SA_modules.append(
                    pointnet2_modules.PointnetSAModuleMSG_SSD(
                        npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                        radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                        nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                        mlps=self.mlps,
                        use_xyz=True,
                        out_channle=self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k],
                        fps_type=self.model_cfg.SA_CONFIG.FPS_TYPE[k],
                        fps_range=self.model_cfg.SA_CONFIG.FPS_RANGE[k],
                        dilated_group=False,
                    )
                )


            elif self.layer_types[k] == 'Vote_Layer':
                self.SA_modules.append(pointnet2_modules.Vote_layer(mlp_list=self.model_cfg.SA_CONFIG.MLPS[k],
                                                                    pre_channel=self.channel_out_list[self.layer_inputs[k]],
                                                                    max_translate_range=self.max_translate_range))

            self.channel_out_list.append(self.model_cfg.SA_CONFIG.AGGREATION_CHANNEL[k])

        self.skip_channel_list = [1, 64, 128, 256, 512]
        self.channel_out = 512
        
        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                points: (num_points, 4 + C), [batch_idx, x, y, z, ...]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        batch_idx, xyz, features = self.break_up_pc(points)  # [b*n], [b*n, 3], [b*n, 1]

        xyz_batch_cnt = xyz.new_zeros(batch_size).int()

        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()
        assert xyz_batch_cnt.min() == xyz_batch_cnt.max()

        xyz = xyz.view(batch_size, -1, 3)  # [b, n, 3]
        features = features.view(batch_size, -1, features.shape[-1]).permute(0, 2, 1) if features is not None else None # [b, 1, n]

        encoder_xyz, encoder_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            xyz_input = encoder_xyz[self.layer_inputs[i]]
            feature_input = encoder_features[self.layer_inputs[i]]
            if self.layer_types[i] == 'SA_Layer':
                ctr_xyz = None
                if self.ctr_indexes[i] != -1:
                    ctr_xyz = encoder_xyz[self.ctr_indexes[i]]
                li_xyz, li_features, fs_idx = self.SA_modules[i](xyz_input, feature_input, ctr_xyz=ctr_xyz)
            elif self.layer_types[i] == 'Vote_Layer':
                li_xyz, li_features, ctr_offsets = self.SA_modules[i](xyz_input, feature_input)
                centers = li_xyz
                centers_origin = xyz_input
            encoder_xyz.append(li_xyz) #[[b, 16384 ,3], [b, 4096, 3], [b ,1024, 3], [b, 512, 3], [b, 256, 3],#[b, 256, 3]vote#, [b ,256, 3]]
            encoder_features.append(li_features) #[[b ,1 ,16384], [b, 64, 4096], [b, 128, 1024], [b, 256, 512], [b, 256, 256], #[b, 128, 256]vote#, [b, 512, 256]]

        ctr_batch_idx = batch_idx.view(batch_size, -1)[:, :ctr_offsets.shape[1]]
        ctr_batch_idx = ctr_batch_idx.contiguous().view(-1)
        batch_dict['ctr_offsets'] = torch.cat((ctr_batch_idx[:, None].float(), ctr_offsets.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers'] = torch.cat((ctr_batch_idx[:, None].float(), centers.contiguous().view(-1, 3)), dim=1)
        batch_dict['centers_origin'] = torch.cat((ctr_batch_idx[:, None].float(), centers_origin.contiguous().view(-1, 3)), dim=1)
        center_features = encoder_features[-1].permute(0, 2, 1).contiguous().view(-1, encoder_features[-1].shape[1])
        batch_dict['centers_features'] = center_features
        batch_dict['ctr_batch_idx'] = ctr_batch_idx  # center各点属于的batch

        return batch_dict

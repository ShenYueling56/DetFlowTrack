import torch
import torch.nn as nn
from ...ops.pointnet2.pointnet2_batch import pointnet2_modules
import torchsnooper
#两层特征聚合,先帧间特征聚合再帧内特征聚合

class SAFusioinLayer(nn.Module):
    def __init__(self, model_cfg, input_channels, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.Inter_SA_modules = nn.ModuleList()
        self.Intra_SA_modules = nn.ModuleList()

        self.channel_in = input_channels
        self.channel_out_list = [self.channel_in]

        print("input channel")
        print(input_channels)

        # inter sa
        for k in range(self.model_cfg.INTER_SA_CONFIG.NPOINTS.__len__()):
            self.channel_in = self.channel_out_list[-1]
            self.mlps = self.model_cfg.INTER_SA_CONFIG.MLPS[k].copy()
            for idx in range(self.mlps.__len__()):
                self.mlps[idx] = [self.channel_in] + self.mlps[idx]
            self.Inter_SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG_SSD(
                    npoint=self.model_cfg.INTER_SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.INTER_SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.INTER_SA_CONFIG.NSAMPLE[k],
                    mlps=self.mlps,
                    use_xyz=True,
                    out_channle=self.model_cfg.INTER_SA_CONFIG.AGGREATION_CHANNEL[k],
                    dilated_group=False,
                )
            )
            self.channel_out_list.append(self.model_cfg.INTER_SA_CONFIG.AGGREATION_CHANNEL[k])

        # intra sa
        for k in range(self.model_cfg.INTRA_SA_CONFIG.NPOINTS.__len__()):
            self.channel_in = self.channel_out_list[-1]
            self.mlps = self.model_cfg.INTRA_SA_CONFIG.MLPS[k].copy()
            for idx in range(self.mlps.__len__()):
                self.mlps[idx] = [self.channel_in] + self.mlps[idx]
                print("intra mlps\n", self.mlps[idx])
            self.Intra_SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG_SSD(
                    npoint=self.model_cfg.INTRA_SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.INTRA_SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.INTRA_SA_CONFIG.NSAMPLE[k],
                    mlps=self.mlps,
                    use_xyz=True,
                    out_channle=self.model_cfg.INTRA_SA_CONFIG.AGGREATION_CHANNEL[k],
                    dilated_group=False,
                )
            )

    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features

    # @torchsnooper.snoop()
    def forward(self, batch_dict1, batch_dict2, center_flow_pred):

        batch_size = batch_dict1['batch_size']
        centers1 = batch_dict1['centers'].clone()
        centers2 = batch_dict2['centers'].clone()
        batch_idx1, xyz1, _ = self.break_up_pc(centers1)
        batch_idx2, xyz2, _ = self.break_up_pc(centers2)
        xyz1 = xyz1.view(batch_size, -1, 3)  # [b, n, 3]
        xyz2 = xyz2.view(batch_size, -1, 3)  # [b, n, 3]
        xyz12 = xyz1 + center_flow_pred.view(batch_size, -1, 3)

        centers_det_features1 = batch_dict1['point_features'].clone()
        centers_det_features1 = centers_det_features1.view(batch_size, -1, centers_det_features1.shape[-1]).permute(0, 2, 1)  # [b, c, n]
        centers_det_features2 = batch_dict2['point_features'].clone()
        centers_det_features2 = centers_det_features2.view(batch_size, -1, centers_det_features2.shape[-1]).permute(0, 2, 1) # [b, c, n]

        encoder_xyz12, encoder_det_feature1 = [xyz12], [centers_det_features1]
        encoder_xyz2, encoder_det_feature2 = [xyz2], [centers_det_features2]

        # inter sa
        for i in range(len(self.Inter_SA_modules)):
            xyz_input12 = encoder_xyz12[-1]
            xyz_input2 = encoder_xyz2[-1]
            xyz_input = torch.cat((xyz_input12, xyz_input2), 1)
            det_feature_input1 = encoder_det_feature1[-1]
            det_feature_input2 = encoder_det_feature2[-1]
            det_feature_input = torch.cat((det_feature_input1, det_feature_input2), 2)
            # 以当前帧的各个center点为中心,聚合前一帧的各个点
            li_xyz, li_det_features, fs_idx = self.Inter_SA_modules[i](xyz_input, det_feature_input,
                                                                               ctr_xyz=xyz_input2)
            encoder_xyz2.append(li_xyz)
            encoder_det_feature2.append(li_det_features)
        # intra sa
        for i in range(len(self.Intra_SA_modules)):
            xyz_input2 = encoder_xyz2[-1]
            det_feature_input2 = encoder_det_feature2[-1]
            li_xyz, li_det_features, fs_idx = self.Intra_SA_modules[i](xyz_input2, det_feature_input2,
                                                                       ctr_xyz=xyz_input2)
            encoder_xyz2.append(li_xyz)
            encoder_det_feature2.append(li_det_features)

        batch_dict2['point_features'] = encoder_det_feature2[-1].permute(0, 2, 1).contiguous().view(-1, encoder_det_feature2[-1].shape[1])

        return batch_dict2








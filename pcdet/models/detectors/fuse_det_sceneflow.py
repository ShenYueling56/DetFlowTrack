### 利用场景流将特征做wrap,联合两帧特征进行检测
import torch
from .det_sceneflow_3d_template import DetSceneflow3DTemplate
import torchsnooper

first_frame=True
last_batch_dict = None
last_seq = 16


class FusionDetSceneflowPoint(DetSceneflow3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_list = self.build_networks()
        self.build_networks()

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
        global last_batch_dict
        global first_frame
        global last_seq
        global last_pred_dicts
        batch_dict1 = {}
        batch_dict2 = {}
        for key, val in batch_dict.items():
            if key.split("_", 1)[0] == "frame1":
                batch_dict1[key.split("_", 1)[1]] = val
                continue
            if key.split("_", 1)[0] == "frame2":
                batch_dict2[key.split("_", 1)[1]] = val
                continue
            batch_dict1[key] = val
            batch_dict2[key] = val

        # backbone3d:
        backbone_layer = self.backbone_3d
        batch_dict1 = backbone_layer(batch_dict1)
        batch_dict2 = backbone_layer(batch_dict2)

        # first frame
        if not batch_dict1['seq_id'][0] == last_seq:
            first_frame = True

        # sceneflow head
        sceneflow_head = self.sceneflow_head
        if self.training | first_frame:
            flow_batch_dict = {
                "frame1": batch_dict1,
                "frame2": batch_dict2
            }
            if first_frame:
                last_batch_dict = batch_dict2
                last_seq = batch_dict1['seq_id'][0]
        else:
            flow_batch_dict = {
                "frame1": last_batch_dict,
                "frame2": batch_dict2,
            }
            last_batch_dict = batch_dict2
            last_seq = batch_dict1['seq_id'][0]
        batch_dict, ret_dict = sceneflow_head(flow_batch_dict)

        # feature wrap and fusion
        B, N = batch_dict['center_flow_pred'].shape[0], batch_dict['center_flow_pred'].shape[2]
        center_flow_pred = batch_dict['center_flow_pred'].reshape(B*N, -1) # B*N, 3

        # if self.training | first_frame:
        #     centers_det_features1 = batch_dict1['centers_det_features'].clone()
        #     centers12 = batch_dict1['centers'].clone()
        #     ctr_offsets1 = batch_dict1['ctr_offsets'].clone()
        #     centers_origin12 = batch_dict1['centers_origin'].clone()
        # else:
        #     centers_det_features1 = last_batch_dict['centers_det_features'].clone()
        #     centers12 = last_batch_dict['centers'].clone()
        #     ctr_offsets1 = last_batch_dict['ctr_offsets'].clone()
        #     centers_origin12 = last_batch_dict['centers_origin'].clone() # B*N, 4

        # # print("centers origin ", centers_origin12.shape)
        # centers12[:, 1:] = centers12[:, 1:] + center_flow_pred
        # centers_origin12[:, 1:] = centers_origin12[:, 1:] + center_flow_pred

        # # 直接将两帧的点进行拼接,没有融合的过程
        # centers2 = torch.cat((batch_dict2['centers'], centers12), 0)
        # centers_origin2 = torch.cat((batch_dict2['centers_origin'], centers_origin12), 0)
        # centers_det_features2 = torch.cat((batch_dict2['centers_det_features'], centers_det_features1), 0)
        # ctr_offsets2 = torch.cat((batch_dict2['ctr_offsets'], ctr_offsets1), 0)
        # batch_dict2['centers'] = centers2
        # batch_dict2['centers_origin'] = centers_origin2
        # batch_dict2['ctr_offsets'] = ctr_offsets2
        # batch_dict2['centers_det_features'] = centers_det_features2

        fusion_layer = self.fusion_layer
        batch_dict2 = fusion_layer(batch_dict1, batch_dict2, center_flow_pred)

        # det head
        det_head = self.point_head
        # if self.training | first_frame:
        #     batch_dict1, ret_dict1 = det_head(batch_dict1)
        batch_dict2, ret_dict2 = det_head(batch_dict2)

        # calculate loss
        if self.training:
            loss, tb_dict = self.get_training_loss(ret_dict2, ret_dict)
            ret_dict = {
                'loss': loss
            }
            # print("loss ", loss.item())
            return ret_dict, tb_dict

        else:
            if first_frame:
                first_frame = False
            else:
                batch_dict1 = last_batch_dict
            batch_dict.update({
                "frame1": batch_dict1,
                "frame2": batch_dict2,
            })
            # print("first frame ", first_frame)
            # print("batch_index ", batch_dict1['batch_index'].shape)
            # print("batch_index ", batch_dict2['batch_index'].shape)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if 'center_flow_labels' in ret_dict:
                sceneflow_loss_point, sceneflow_tb_dict = self.sceneflow_head.get_loss(ret_dict)  # 获得场景流loss
                recall_dicts.update({
                    'sceneflow_loss': sceneflow_loss_point
                })
            last_batch_dict = batch_dict2
            return pred_dicts, recall_dicts

    def get_training_loss(self, ret_dict2, ret_dict):
        det_loss_point, tb_dict = self.point_head.get_loss(ret_dict2) # 只对当前帧计算检测loss
        sceneflow_loss_point, sceneflow_tb_dict = self.sceneflow_head.get_loss(ret_dict) # 获得场景流loss
        weight = self.model_cfg.LOSS.DET_FLOW_WEIGHT
        loss = det_loss_point*weight[0] + sceneflow_loss_point*weight[1]
        # print("det_loss_point ", str(det_loss_point), " flow_loss_point ", sceneflow_loss_point)
        loss = loss / (weight[0]+weight[1])
        # print("loss ", loss, " weight0 ", weight[0], " weight1 ", weight[1])
        tb_dict.update(sceneflow_tb_dict)
        # loss = sceneflow_loss_point
        # tb_dict = sceneflow_tb_dict
        return loss, tb_dict
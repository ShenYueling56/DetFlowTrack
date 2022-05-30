from .det_sceneflow_3d_template import DetSceneflow3DTemplate

first_frame=True
last_batch_dict = None
last_seq = 16
import torchsnooper
class DetSceneflowPoint(DetSceneflow3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_list = self.build_networks()
        self.build_networks()

    # @torchsnooper.snoop()
    def forward(self, batch_dict):
        global last_batch_dict
        global first_frame
        global last_seq
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
        # backbone_layer = self.module_list[0]
        backbone_layer = self.backbone_3d
        batch_dict1 = backbone_layer(batch_dict1)
        batch_dict2 = backbone_layer(batch_dict2)
        # det head
        # det_head = self.module_list[1]
        det_head = self.point_head
        batch_dict1, ret_dict1 = det_head(batch_dict1)
        batch_dict2, ret_dict2 = det_head(batch_dict2)

        # sceneflow
        # sceneflow_head = self.module_list[2]
        if not batch_dict1['seq_id'] == last_seq:
            first_frame = True
        sceneflow_head = self.sceneflow_head
        if self.training | first_frame:
            flow_batch_dict = {
                "frame1": batch_dict1,
                "frame2": batch_dict2,
            }
            if first_frame:
                last_batch_dict = batch_dict2
                first_frame = False
                last_seq = batch_dict1['seq_id']
        else:
            flow_batch_dict = {
                "frame1": last_batch_dict,
                "frame2": batch_dict2,
            }
            last_batch_dict = batch_dict2
            last_seq = batch_dict1['seq_id']
        batch_dict, ret_dict = sceneflow_head(flow_batch_dict)
        if self.training:
            loss, tb_dict = self.get_training_loss(ret_dict2, ret_dict)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict

        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            if 'center_flow_labels' in ret_dict:
                sceneflow_loss_point, sceneflow_tb_dict = self.sceneflow_head.get_loss(ret_dict)  # 获得场景流loss
                recall_dicts.update({
                    'sceneflow_loss': sceneflow_loss_point
                })
            return pred_dicts, recall_dicts

    def get_training_loss(self, ret_dict2, ret_dict):
        det_loss_point, tb_dict = self.point_head.get_loss(ret_dict2) # 获得检测loss
        sceneflow_loss_point, sceneflow_tb_dict = self.sceneflow_head.get_loss(ret_dict) # 获得场景流loss
        weight = self.model_cfg.LOSS.DET_FLOW_WEIGHT
        loss = det_loss_point*weight[0] + sceneflow_loss_point*weight[1]
        loss = loss / (weight[0]+weight[1])
        tb_dict.update(sceneflow_tb_dict)
        # loss = sceneflow_loss_point
        # tb_dict = sceneflow_tb_dict
        return loss, tb_dict
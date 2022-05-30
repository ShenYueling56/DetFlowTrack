from .det_sceneflow_3d_template import DetSceneflow3DTemplate
first_frame=True
last_batch_dict = None
last_seq = 16

class PVRCNNFD(DetSceneflow3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        # for cur_module in self.module_list:
        #     batch_dict = cur_module(batch_dict)
        # print("batch dict keys\n", batch_dict.keys())
        global last_batch_dict
        global first_frame
        global last_seq
        # 分离tracking数据集前一帧和当前帧的数据
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

        # 判断是否是seq的初始frame
        if not batch_dict1['seq_id'][0].item() == last_seq:
            first_frame = True

        # 运行检测网络
        for cur_module in self.module_list:
            from pcdet.models.sceneflow_heads.flow_point_head import FlowPointHead
            from pcdet.models.roi_heads.pvrcnn_head import PVRCNNHead
            from pcdet.models.fusion_layers import SAFusioinLayer
            if(cur_module.__class__ in [FlowPointHead, PVRCNNHead, SAFusioinLayer]):
                continue
            from pcdet.models.dense_heads.anchor_head_single import AnchorHeadSingle
            from pcdet.models.dense_heads.point_head_simple import PointHeadSimple

            if(cur_module.__class__ == AnchorHeadSingle):
                if self.training | first_frame:
                    batch_dict1, ret_dict_anchorhead1 = cur_module(batch_dict1)
                batch_dict2, ret_dict_anchorhead2 = cur_module(batch_dict2)
            elif(cur_module.__class__ == PointHeadSimple):
                if self.training | first_frame:
                    batch_dict1, ret_dict_pointhead1 = cur_module(batch_dict1)
                batch_dict2, ret_dict_pointhead2 = cur_module(batch_dict2)
            else:
                if self.training | first_frame:
                    batch_dict1 = cur_module(batch_dict1)
                batch_dict2 = cur_module(batch_dict2)

        # 运行场景流网络
        # todo:将检测作用到场景流学习:'batch_point_box_preds' 'batch_point_cls_preds' 待验证
        if self.training | first_frame:
            # keypoints坐标:'point_coords' (b*n, 4)
            # keypoints 直接拼接的特征:'point_features_before_fusion' (b*n, C) 用于前景点概率预测
            # keypoints mlp之后的特征:'point_features' (b*n, C)  用于检测和场景流学习
            batch_dict1['centers'] = batch_dict1['point_coords']
            batch_dict2['centers'] = batch_dict2['point_coords']
            batch_dict1['centers_flow_features'] = batch_dict1['flow_point_feature']
            batch_dict2['centers_flow_features'] = batch_dict2['flow_point_feature']
            flow_batch_dict = {
                "frame1": batch_dict1,
                "frame2": batch_dict2
            }
            if first_frame:
                last_batch_dict = batch_dict2
                first_frame = False
                last_seq = batch_dict1['seq_id'][0]
        else:
            # keypoints坐标:'point_coords' (b*n, 4)
            # keypoints 直接拼接的特征:'point_features_before_fusion' (b*n, C) 用于前景点概率预测
            # keypoints mlp之后的特征:'point_features' (b*n, C)  用于检测和场景流学习
            batch_dict2['centers'] = batch_dict2['point_coords']
            batch_dict2['centers_flow_features'] = batch_dict2['flow_point_feature']
            flow_batch_dict = {
                "frame1": last_batch_dict,
                "frame2": batch_dict2,
            }
            batch_dict1 = last_batch_dict
            last_batch_dict = batch_dict2
            last_seq = batch_dict1['seq_id'][0]

        batch_dict, ret_dict_flow = self.sceneflow_head(flow_batch_dict)

        # 多帧融合网络
        B, N = batch_dict['center_flow_pred'].shape[0], batch_dict['center_flow_pred'].shape[2]
        center_flow_pred = batch_dict['center_flow_pred'].reshape(B * N, -1)  # B*N, 3
        print("det feature shape")
        print(batch_dict1['point_features'].shape)
        batch_dict2 = self.fusion_layer(batch_dict1, batch_dict2, center_flow_pred)

        # 检测头
        if self.training | first_frame:
            batch_dict1, ret_dict_pvrcnnhead1 = self.roi_head(batch_dict1)
        batch_dict2, ret_dict_pvrcnnhead2 = self.roi_head(batch_dict2)

        if self.training:
            loss, tb_dict = self.get_training_loss(ret_dict_anchorhead2, ret_dict_pointhead2, ret_dict_pvrcnnhead2, ret_dict_flow)

            loss_dict = {
                'loss': loss
            }
            return loss_dict, tb_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, ret_dict_anchorhead, ret_dict_pointhead, ret_dict_pvrcnnhead, ret_dict_flow):
        loss_rpn, tb_dict = self.dense_head.get_loss(ret_dict_anchorhead)
        loss_point, tb_dict = self.point_head.get_loss(ret_dict_pointhead, tb_dict)
        loss_rcnn, tb_dict = self.roi_head.get_loss(ret_dict_pvrcnnhead, tb_dict)
        loss_flow, tb_dict = self.sceneflow_head.get_loss(ret_dict_flow, tb_dict)
        loss = loss_rpn + loss_point + loss_rcnn + loss_flow
        return loss, tb_dict

from .detector3d_template import Detector3DTemplate


class Point3DSSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for i, cur_module in enumerate(self.module_list):
            if i == 1:
                batch_dict, forward_dict = cur_module(batch_dict)
            else:
                batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict = self.get_training_loss(forward_dict)

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self, forward_dict):
        loss_point, tb_dict = self.point_head.get_loss(forward_dict)

        loss = loss_point
        return loss, tb_dict

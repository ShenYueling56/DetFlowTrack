from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .point_3DSSD import Point3DSSD
from .point_multi_scale_3DSSD import MultiScalePoint3DSSD
from .det_sceneflow import DetSceneflowPoint
from .solo_det_sceneflow import SoloDetSceneflowPoint
from .fuse_det_sceneflow import FusionDetSceneflowPoint
from .pv_rcnn_det_track import PVRCNNDT
from .pv_rcnn_fuse_det import PVRCNNFD

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    '3DSSD': Point3DSSD,
    'multi_scale_3DSSD': MultiScalePoint3DSSD,
    '3DSSDT': DetSceneflowPoint,
    'SOLO_3DSSDT': SoloDetSceneflowPoint,
    'Fusion3DSSDT': FusionDetSceneflowPoint,
    'PVRCNNDT': PVRCNNDT,
    'PVRCNNFD': PVRCNNFD
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model

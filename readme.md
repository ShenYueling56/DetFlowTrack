# DetFlowTrack: 3D Multi-object Tracking based on Simultaneous Optimization of Object Detection and Scene Flow Estimation

Created by Yueling Shen, Guangming Wang, and Hesheng Wang

### Citation
If you find our work useful in your research, please cite:
```
@article{shen2021detflowtrack,
  title={DetFlowTrack: 3D Multi-object Tracking based on Simultaneous Optimization of Object Detection and Scene Flow Estimation},
  author={Shen, Yueling and Wang, Guangming and Wang, Hesheng},
  journal={arXiv preprint  	arXiv:2203.02157},
  year={2022}
}
```
### Abstract
3D Multi-Object Tracking (MOT) is an important part of the unmanned vehicle perception module. Most methods optimize object detection and data association independently. These methods make the network structure complicated and limit the improvement of MOT accuracy. 
we proposed a 3D MOT framework based on simultaneous optimization of object detection and scene flow estimation. In the framework, a detection-guidance scene flow module is proposed to relieve the problem of incorrect inter-frame assocation. For more accurate scene flow label especially in the case of motion with rotation, a box-transformation-based scene flow ground truth calculation method is proposed. 
Experimental results on the KITTI
MOT dataset show competitive results over the state-of-the-arts and the robustness under extreme motion with rotation.

### Usage

#### 1. Installation
1.1 Install the python dependencies.
```
pip install -r requirements.txt
```
1.2 Install according to the [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).

#### 2. Datasets Preparation
Generate Object Database with KITTI Object DataSets.
```
python pcdet/datasets/kitti/kitti_dataset.py create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

Preprocess dataset: KITTI MOT Datasets
```
python pcdet/datasets/kitti_tracking/kitti_tracking_dataset.py create_kitti_infos tools/cfgs/dataset_configs/kitti_tracking_dataset_voxel.yam
```
#### 3. Train
```
cd tools
python train.py --cfg_file=cfgs/kitti_models/pv_rcnn_DetTrack.yaml
```

#### 4. Eval
```
cd tools
python inference.py --ckpt=../pretrained_weight/checkpoint_epoch_76.pth --cfg_file=cfgs/kitti_models/pv_rcnn_DetTrack.yaml
```
在cfg中指定项目目录ROOT_DIR, 最终结果保存在"ROOT_DIR/output/"下.多目标跟踪结果保存在track_results/data中.
可通过[TrackEval](https://github.com/JonathonLuiten/TrackEval) 得到HOTA指标
提供的预训练权重pretrained_weight/checkpoint_epoch_76.pth(链接：https://pan.baidu.com/s/1e_klYYTPfxwaYy9beyiBdQ 
提取码：axcc )的精度指标为:

| 类别 | HOTA | DetA | AssA | LocA |
| ------ | ------ | ------ | ------| ------|
| Car | 76.685 | 72.837 | 80.988 | 89.632 |
| Pedestrian | 45.133 | 41.439 | 49.248 | 73.218 |

#### 5. Test
```
cd tools
python inference.py --ckpt=../pretrained_weight/checkpoint_epoch_76.pth --cfg_file=cfgs/kitti_models/pv_rcnn_DetTrack_test.yaml
```
将跟踪结果提交到KITTI MOT服务器上,结果为

![image](https://github.com/IRMVLab/DetFlowTrack/blob/main/test_result/result.PNG)


### Acknowledgments
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)

[TrackEval](https://github.com/JonathonLuiten/TrackEval)
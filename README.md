# AS-MLP for Object Detection

This repo contains the supported code and configuration files to reproduce object detection results of [AS-MLP](https://arxiv.org/pdf/2107.08391.pdf). It is based on [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection).


## Results and Models

### Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | Params | FLOPs | config | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| AS-MLP-T | ImageNet-1K | 1x | 44.0 | 40.0 | 48M | 260G | [config](configs/asmlp/mask_rcnn_asmlp_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py) |  |
| AS-MLP-T | ImageNet-1K | 3x | 46.0 | 41.5 | 48M | 260G | [config](configs/asmlp/mask_rcnn_asmlp_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py) |  |
| AS-MLP-S | ImageNet-1K | 1x | 46.7 | 42.0 | 69M | 346G | [config](configs/asmlp/mask_rcnn_asmlp_small_patch4_window7_mstrain_480-800_adamw_1x_coco.py) |  |
| AS-MLP-S | ImageNet-1K | 3x | 47.8 | 42.9 | 69M | 346G | [config](configs/asmlp/mask_rcnn_asmlp_small_patch4_window7_mstrain_480-800_adamw_3x_coco.py) |  |

### Cascade Mask R-CNN

| Backbone | Pretrain | Lr Schd | box mAP | mask mAP | Params | FLOPs | config  | model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| AS-MLP-T | ImageNet-1K | 1x | 48.4 | 42.0 | 86M | 739G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) |  |
| AS-MLP-T | ImageNet-1K | 3x | 50.1 | 43.5 | 86M | 739G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) | |
| AS-MLP-S | ImageNet-1K | 1x | 50.5 | 43.7 | 107M | 824G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) |  |
| AS-MLP-S | ImageNet-1K | 3x | 51.1 | 44.2 | 107M | 824G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_small_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) |  |
| AS-MLP-B | ImageNet-1K | 1x | 51.1 | 44.2 | 145M | 961G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_1x_coco.py) |  |
| AS-MLP-B | ImageNet-1K | 3x | 51.5 | 44.7 | 145M | 961G | [config](configs/asmlp/cascade_mask_rcnn_asmlp_base_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py) |  |


**Notes**: 

- **Pre-trained models can be downloaded from [AS-MLP for ImageNet Classification](https://github.com/svip-lab/AS-MLP)**.


## Usage

### Installation

Please refer to [get_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

### Inference
```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm

# multi-gpu testing
tools/dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval bbox segm
```

### Training

To train a detector with pre-trained models, run:
```
# single-gpu training
python tools/train.py <CONFIG_FILE> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments]

# multi-gpu training
tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> --cfg-options model.pretrained=<PRETRAIN_MODEL> [model.backbone.use_checkpoint=True] [other optional arguments] 
```
For example, to train a Mask R-CNN model with a `AS-MLP-T` backbone and 8 gpus, run:
```
tools/dist_train.sh configs/asmlp/mask_rcnn_asmlp_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco.py 8 --cfg-options model.pretrained=<PRETRAIN_MODEL> 
```

**Note:** `use_checkpoint` is used to save GPU memory. Please refer to [this page](https://pytorch.org/docs/stable/checkpoint.html) for more details.


### Apex (optional):
We use apex for mixed precision training by default. To install apex, run:
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Citation
```
@article{Lian_2021_ASMLP,
  author = {Lian, Dongze and Yu, Zehao and Sun, Xing and Gao, Shenghua},
  title = {AS-MLP: An Axial Shifted MLP Architecture for Vision},
  journal={ICLR},
  year = {2022}
}
```

## Other Links

> **Image Classification**: See [AS-MLP for Image Classification](https://github.com/svip-lab/AS-MLP).




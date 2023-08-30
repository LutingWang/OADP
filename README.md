# Object-Aware Distillation Pyramid

```text
     _/_/      _/_/    _/_/_/    _/_/_/
  _/    _/  _/    _/  _/    _/  _/    _/
 _/    _/  _/_/_/_/  _/    _/  _/_/_/
_/    _/  _/    _/  _/    _/  _/
 _/_/    _/    _/  _/_/_/    _/
```

This repository is the official implementation of "[Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection](https://arxiv.org/abs/2303.05892)".

[![lint](https://github.com/LutingWang/OADP/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/OADP/actions/workflows/lint.yaml)

## Installation

Create a conda environment and activate it.

```shell
conda create -n oadp python=3.10
conda activate oadp
```

Install `PyTorch` following the [official documentation](https://pytorch.org/).
For example,

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```

Install `MMDetection` following the [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
For example,

```bash
pip install openmim
mim install mmcv_full==1.7.0
pip install mmdet==2.25.2
```

Install other dependencies.

```bash
pip install todd_ai==0.3.0
pip install git+https://github.com/LutingWang/CLIP.git
pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pip install nni scikit-learn==1.1.3
```

## Preparation

### Datasets

Download the [MS-COCO](https://cocodataset.org/#download) dataset to `data/coco`.

```text
OADP/data/coco
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017
│   └── ...
└── val2017
    └── ...
```

Download the [LVIS v1.0](https://www.lvisdataset.org/dataset) dataset to `data/lvis_v1`.

```text
OADP/data/lvis_v1
├── annotations
│   ├── lvis_v1_train.json
│   └── lvis_v1_val.json
├── train2017 -> ../coco/train2017
│   └── ...
└── val2017 -> ../coco/train2017
    └── ...
```

### Annotations

```bash
python -m oadp.build_annotations
```

The following files will be generated

```text
OADP/data
├── coco
│   └── annotations
│       ├── instances_train2017.48.json
│       ├── instances_train2017.65.json
│       ├── instances_val2017.48.json
│       ├── instances_val2017.65.json
│       └── instances_val2017.65.min.json
└── lvis_v1
    └── annotations
        ├── lvis_v1_train.1203.json
        ├── lvis_v1_train.866.json
        ├── lvis_v1_val.1203.json
        └── lvis_v1_val.866.json
```

### Pretrained Models

Download the CLIP model.

```shell
python -c "import clip; clip.load_default()"
```

Download the ResNet50 model.

```shell
mkdir pretrained
python -c "import torchvision; _ = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.get_state_dict(True)"
ln -s ~/.cache/torch/hub/checkpoints/ pretrained/torchvision
```

Download and rename `soco_star_mask_rcnn_r50_fpn_400e.pth` from [Baidu Netdisk](https://pan.baidu.com/s/1FHN-9vsH16w4TAusyHnXvg?pwd=kwps) or [Google Drive](https://drive.google.com/file/d/1z6Tb2MPFJDv9qpEyn_J0cJcXOguKTiL0/view?usp=sharing).

Download the [DetPro][] prompt from [Baidu Netdisk](https://pan.baidu.com/s/1MjV1DqiO0gHftyKjuiPrTA?pwd=uvab).

Organize the pretrained models as follows

```text
OADP/pretrained
├── clip
│   └── ViT-B-32.pt
├── detpro
│   └── iou_neg5_ens.pth
├── torchvision
│   └── resnet50-0676ba61.pth
└── soco
    └── soco_star_mask_rcnn_r50_fpn_400e.pth
```

### Prompts

Generate the ViLD prompts.

```bash
python -m oadp.prompts.vild
```

Download `ml_coco.pth` from [Baidu Netdisk][].

Generate the [DetPro][] prompts.

```bash
python -m oadp.prompts.detpro
```

Organize the prompts as follows

```text
OADP/data/prompts
├── vild.pth
└── ml_coco.pth
```

### Proposals

Download the proposals from [Baidu Netdisk][].

Organize the proposals as follows

```text
OADP/data
├── coco
│   └── proposals
│       ├── rpn_r101_fpn_coco_train.pkl
│       ├── rpn_r101_fpn_coco_val.pkl
│       ├── oln_r50_fpn_coco_train.pkl
│       └── oln_r50_fpn_coco_val.pkl
└── lvis_v1
    └── proposals
        ├── oln_r50_fpn_lvis_train.pkl
        └── oln_r50_fpn_lvis_val.pkl
```

## OADP

Most commands listed in this section supports the `DRY_RUN` mode.
When the `DRY_RUN` environment variable is set to `True`, the command that follows will not execute the time-consuming parts.
This functionality is intended for quick integrity check.

Most commands run on both CPU and GPU servers.
For CPU, use the `python` command.
For GPU, use the `torchrun` command.
Do not use `python` on GPU servers, since the command will attempt to initialize distributed training.

For all commands listed in this section, `[...]` means optional parts and `(...|...)` means choices.
For example,

```shell
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS})
```

is equivalent to the following four possible commands

```shell
DRY_RUN=True torchrun --nproc_per_node=${GPUS}  # GPU under the DRY_RUN mode
DRY_RUN=True python                             # CPU under the DRY_RUN mode
torchrun --nproc_per_node=${GPUS}               # GPU
python                                          # CPU
```

### OAKE

The following scripts extract features with CLIP, which can be very time-consuming. Therefore, all the scripts support automatically resuming, by skipping existing feature files. However, the existing feature files are sometimes broken. In such cases, users can set the `auto_fix` option to inspect the integrity of each feature file.

Extract globals and blocks features, which can be used for both coco and lvis

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.globals oake/globals configs/oake/globals.py [--override .train.dataloader.dataset.auto_fix:True .val.dataloader.dataset.auto_fix:True]
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.blocks oake/blocks configs/oake/blocks.py [--override .train.dataloader.dataset.auto_fix:True .val.dataloader.dataset.auto_fix:True]
```

Extract objects features for coco

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.objects oake/objects configs/oake/objects_coco.py [--override .train.dataloader.dataset.auto_fix:True .val.dataloader.dataset.auto_fix:True]
```

Extract objects features for lvis

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.objects oake/objects configs/oake/objects_lvis.py [--override .train.dataloader.dataset.auto_fix:True .val.dataloader.dataset.auto_fix:True]
```

Feature extraction can be very time consuming.
Therefore, we provide archives of the extracted features on [Baidu Netdisk][].
The extracted features are archived with the following command

```bash
cd data/coco/oake/

tar -zcf globals.tar.gz globals
tar -zcf blocks.tar.gz blocks
tar -zcf objects.tar.gz objects/val2017

cd objects/train2017
ls > objects
split -d -3000 - objects. < objects
for i in objects.[0-9][0-9]; do
    zip -q -9 "$i.zip" -@ < "$i"
    mv "$i.zip" ../..
done
rm objects*
```

The final directory for OAKE should look like

```text
OADP/data
├── coco
│   └── oake
│       ├── blocks
│       │   ├── train2017
│       │   └── val2017
│       ├── globals
│       │   ├── train2017
│       │   └── val2017
│       └── objects
│           ├── train2017
│           └── val2017
└── lvis_v1
    └── oake
        ├── blocks -> ../coco/oake/blocks
        ├── globals -> ../coco/oake/globals
        └── objects
            ├── train2017
            └── val2017
```

### DP

To conduct training for coco

```bash
[DRY_RUN=True] [TRAIN_WITH_VAL_DATASET=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.train vild_ov_coco configs/dp/vild_ov_coco.py [--override .validator.dataloader.dataset.ann_file::data/coco/annotations/instances_val2017.48.json]
[DRY_RUN=True] [TRAIN_WITH_VAL_DATASET=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.train oadp_ov_coco configs/dp/oadp_ov_coco.py [--override .validator.dataloader.dataset.ann_file::data/coco/annotations/instances_val2017.48.json]
```

To conduct training for lvis

```bash
[DRY_RUN=True] [TRAIN_WITH_VAL_DATASET=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.train oadp_ov_lvis configs/dp/oadp_ov_lvis.py
```

To test a specific checkpoint (use --metrics=segm to enable instance segmentation test, metrics is set to bounding box by default)

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.test configs/dp/oadp_ov_lvis.py work_dirs/oadp_ov_lvis/epoch_24.pth
```

NNI is supported but unnecessary.

```bash
DUMP=work_dirs/dump (python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.test configs/dp/oadp_ov_coco.py work_dirs/oadp_ov_coco/iter_32000.pth
DUMP=work_dirs/dump python tools/nni_dp_test.py
```

## Results

The checkpoints for OADP are available on [Baidu Netdisk][].

### OV COCO

| mAPN50    | Config                                        | Checkpoint                            |
| :-:       | :-:                                           | :-:                                   |
| $31.3$    | [oadp_ov_coco.py](configs/dp/oadp_ov_coco.py) | work_dirs/oadp_ov_coco/iter_32000.pth |

### OV LVIS

| APr       | Config                                        | Checkpoint                            |
| :-:       | :-:                                           | :-:                                   |
| $20.7$    | [oadp_ov_lvis.py](configs/dp/oadp_ov_lvis.py) | work_dirs/oadp_ov_lvis/epoch_24.pth   |

[Baidu Netdisk]: https://pan.baidu.com/s/1HXWYSN9Vk6yDhjRI19JrfQ?pwd=OADP
[DetPro]: https://github.com/dyabel/detpro

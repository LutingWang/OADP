# Object-Aware Distillation Pyramid

```text
     _/_/      _/_/    _/_/_/    _/_/_/
  _/    _/  _/    _/  _/    _/  _/    _/
 _/    _/  _/_/_/_/  _/    _/  _/_/_/
_/    _/  _/    _/  _/    _/  _/
 _/_/    _/    _/  _/_/_/    _/
```

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
pip install lvis scikit-learn==1.1.3
```

> Note that the `requirements.txt` is not intended for users. Please follow the above instructions.

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

The following files will be generated:

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

```text
OADP/pretrained
├── clip
│   └── ViT-B-32.pt
├── torchvision
│   └── resnet50-0676ba61.pth
└── soco
    └── soco_star_mask_rcnn_r50_fpn_400e.pth
```

### Prompts

```bash
mkdir data/prompts
python -m oadp.prompts.vild
```

### Proposals

## OAKE

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.globals oake/globals configs/oake/globals.py
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.blocks oake/blocks configs/oake/blocks.py
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.objects oake/objects configs/oake/objects.py
```

```bash
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.globals oake/globals configs/oake/lvis/globals.py
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.blocks oake/blocks configs/oake/lvis/blocks.py
[DRY_RUN=True] (python|torchrun --nproc_per_node=${GPUS}) -m oadp.oake.objects oake/objects configs/oake/lvis/objects.py
```
## Train

Please assure you have completed the OAKE module above before executing this command, which is a prerequisite.

```bash
(python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.train oadp_ov_coco configs/dp/oadp_ov_coco.py [--override .validator.dataloader.dataset.ann_file::data/coco/annotations/instances_val2017.48.json]
```

```bash
(python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.train oadp_ov_coco configs/dp/oadp_ov_lvis.py [--override .validator.dataloader.dataset.ann_file::data/lvis_v1/annotations/lvis_v1_val.866.json]
```

## Inference

```bash
(python|torchrun --nproc_per_node=${GPUS}) -m oadp.dp.test configs/dp/oadp_ov_coco.py work_dirs/oadp_ov_coco/iter_32000.pth
```

## NNI

```bash
DUMP=work_dirs/dump python tools/nni_dp_test.py
```

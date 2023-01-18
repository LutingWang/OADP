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

Install `MMDetection` following the [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
For example,

```bash
pip install torch torchvision
pip install -U openmim
mim install mmcv_full==1.7.0
pip install mmdet==2.25.2
```

Install other dependencies.

```bash
pip install todd_ai==0.2.4 -i https://pypi.org/simple
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
python tools/build_annotations.py
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
└── clip
    └── ViT-B-32.pt

```

### Prompts

```bash
mkdir data/prompts
python -m oadp.prompts.vild
```

## Training

### OAKE

```bash
python -m oadp.oake.globals_ oake/globals_ configs/oake/globals_.py
```

## Inference

```bash
# CPU
python -m oadp.test configs/dp/object_block_global_coco_48_17.py work_dirs/object_block_global_coco_48_17/iter_32000.pth

# GPU
torchrun --nproc_per_node=${GPUS} -m oadp.test configs/dp/object_block_global_coco_48_17.py work_dirs/object_block_global_coco_48_17/iter_32000.pth

# ODPS
odpsrun
```

# Object-Aware Distillation Pyramid

```text
     _/_/      _/_/    _/_/_/    _/_/_/
  _/    _/  _/    _/  _/    _/  _/    _/
 _/    _/  _/_/_/_/  _/    _/  _/_/_/
_/    _/  _/    _/  _/    _/  _/
 _/_/    _/    _/  _/_/_/    _/
```

[![lint](https://github.com/LutingWang/OADP/actions/workflows/lint.yaml/badge.svg)](https://github.com/LutingWang/OADP/actions/workflows/lint.yaml)

## Preparation

The directory tree should be like this

```text
OADP
├── data
│   ├── coco -> ~/Developer/datasets/coco
│   │   ├── annotations
│   │   │   ├── instances_val2017.json.COCO_48_17.filtered
│   │   │   └── ...
│   │   ├── train2017
│   │   │   └── ...
│   │   └── val2017
│   │       └── ...
│   └── prompts
│       ├── ml_coco.pth
│       ├── vild.pth
│       └── ...
└── ...
```

### Datasets

Download the [MS-COCO](https://cocodataset.org/#download) dataset to `data/coco`.

### Annotations

### Prompt

## Installation

Create a conda environment and activate it.

```shell
conda create -n oadp python=3.8
conda activate oadp
```

Install `MMDetection` following the [official instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md/#Installation).
For example,

```bash
pip install torch==1.9.1+cu102 torchvision==0.10.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U openmim
mim install mmcv_full==1.4.6
pip install mmdet==2.25.2
```

Install other dependencies.

```bash
pip install todd_ai==0.2.4a5 -i https://pypi.org/simple
pip install scikit-learn==1.1.3
```

> Note that the `requirements.txt` is not intended for users. Please follow the above instructions.

## Inference

```bash
# CPU
python tools/test.py configs/dp/object_block_global_coco_48_17.py work_dirs/object_block_global_coco_48_17/iter_32000.pth

# GPU
torchrun --nproc_per_node=${GPUS} tools/test.py configs/dp/object_block_global_coco_48_17.py work_dirs/object_block_global_coco_48_17/iter_32000.pth --launch pytorch

# ODPS
odpsrun
```

## Developer Guides

### Setup

```bash
pip install https://download.pytorch.org/whl/cpu/torch-1.9.1-cp38-none-macosx_11_0_arm64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.10.0-cp38-cp38-macosx_11_0_arm64.whl
pip install -e ./../mmcv
pip install mmdet==2.20
```

```bash
conda install grpcio -c conda-forge
pip install -U todd_ai\[dev,doc,pre-commit,test\]
```

```bash
pre-commit install
pre-commit install -t commit-msg
```

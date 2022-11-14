# Object-Aware Distillation Pyramid

```text
     _/_/      _/_/    _/_/_/    _/_/_/
  _/    _/  _/    _/  _/    _/  _/    _/
 _/    _/  _/_/_/_/  _/    _/  _/_/_/
_/    _/  _/    _/  _/    _/  _/
 _/_/    _/    _/  _/_/_/    _/
```

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

Install `todd`.

```bash
pip install todd_ai==0.2.4a3 -i https://pypi.org/simple
```

> Note that the `requirements.txt` is not intended for users. Please follow the above instructions.

## Developer Guides

### Setup

```bash
pip install https://download.pytorch.org/whl/cpu/torch-1.9.1-cp38-none-macosx_11_0_arm64.whl
pip install https://download.pytorch.org/whl/cpu/torchvision-0.10.0-cp38-cp38-macosx_11_0_arm64.whl
pip install -e ./../mmcv
pip install mmdet==2.20
```

```bash
pip install commitizen
pip install -U pre-commit
pre-commit install
pre-commit install -t commit-msg
```

```bash
pip install coverage pytest
```

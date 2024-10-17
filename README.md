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

```shell
bash setup.sh
```

## Data

<https://toddai.readthedocs.io/en/latest/data/ms_coco.html>

<https://toddai.readthedocs.io/en/latest/data/lvis.html>

```bash
python tools/build_annotations.py
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
└── lvis
    └── annotations
        ├── lvis_train.1203.json
        ├── lvis_train.866.json
        ├── lvis_val.1203.json
        └── lvis_val.866.json
```

## Pretrained Models

<https://toddai.readthedocs.io/en/latest/data/wordnet.html>

<https://toddai.readthedocs.io/en/latest/pretrained/clip.html>

<https://toddai.readthedocs.io/en/latest/pretrained/ram.html>

```shell
mkdir -p pretrained
python -c "import torchvision; _ = torchvision.models.ResNet50_Weights.IMAGENET1K_V1.get_state_dict()"
ln -s ~/.cache/torch/hub/checkpoints/ pretrained/torch
```

## Prompts

```bash
wget https://huggingface.co/datasets/xinyu1205/recognize-anything-plus-model-tag-descriptions/resolve/main/ram_tag_list_4585_llm_tag_descriptions.json -P oadp/prompts/utils/ram
wget https://raw.githubusercontent.com/OPPOMKLab/recognize-anything/main/datasets/openimages_rare_200/openimages_rare_200_llm_tag_descriptions.json -P oadp/prompts/utils/ram
```

```bash
auto_torchrun -m oadp.prompts.val lvis_clip --config type::LVISPrompter --model type::CLIP
auto_torchrun -m oadp.prompts.val lvis_t5 --config type::LVISPrompter --model type::T5
```

Download `ml_coco.pth` from [Baidu Netdisk][].

```text
OADP/data/prompts
└── ml_coco.pth
```

```bash
mkdir -p pretrained/detpro
bypy downfile iou_neg5_ens.pth pretrained/detpro
python -m oadp.prompts.detpro
```

```bash
mkdir -p pretrained/soco
bypy downfile current_mmdetection_Head.pth pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth
```

## OAKE

The following scripts extract features with CLIP, which can be very time-consuming. Therefore, all the scripts support automatically resuming, by skipping existing feature files. However, the existing feature files are sometimes broken. In such cases, users can set the `auto_fix` option to inspect the integrity of each feature file.

Extract globals and blocks features, which can be used for both coco and lvis

```bash
auto_torchrun -m oadp.oake.val oake/coco/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/coco/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/coco/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/lvis/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::LVIS [--auto-fix]

auto_torchrun -m oadp.oake.val oake/coco/dino_globals_cuda configs/oake/dino_globals_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/coco/dino_blocks_cuda configs/oake/dino_blocks_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/coco/dino_objects_cuda configs/oake/dino_objects_cuda.py --config-options dataset::COCO [--auto-fix]
auto_torchrun -m oadp.oake.val oake/lvis/dino_objects_cuda configs/oake/dino_objects_cuda.py --config-options dataset::LVIS [--auto-fix]

auto_torchrun -m oadp.oake.val oake/coco/ram_cuda configs/oake/ram_cuda.py --config-options dataset::COCO
```

```bash
auto_torchrun -m oadp.oake.val oake/objects365/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::Objects365 [--auto-fix]
auto_torchrun -m oadp.oake.val oake/objects365/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::Objects365 [--auto-fix]
auto_torchrun -m oadp.oake.val oake/objects365/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::Objects365 [--auto-fix]
```

```bash
auto_torchrun -m oadp.oake.val oake/v3det/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::V3Det [--auto-fix]
auto_torchrun -m oadp.oake.val oake/v3det/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::V3Det [--auto-fix]
auto_torchrun -m oadp.oake.val oake/v3det/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::V3Det [--auto-fix]
```

```bash
mkdir -p work_dirs/oake/v3det/clip_globals_cuda/output
cp -r work_dirs/oake/v3det/clip_globals_cuda_train/output/* work_dirs/oake/v3det/clip_globals_cuda/output
cp -r work_dirs/oake/v3det/clip_globals_cuda_val/output/* work_dirs/oake/v3det/clip_globals_cuda/output

mkdir -p work_dirs/oake/v3det/clip_blocks_cuda/output
cp -r work_dirs/oake/v3det/clip_blocks_cuda_train/output/* work_dirs/oake/v3det/clip_blocks_cuda/output
cp -r work_dirs/oake/v3det/clip_blocks_cuda_val/output/* work_dirs/oake/v3det/clip_blocks_cuda/output

mkdir -p work_dirs/oake/v3det/clip_objects_cuda/output
cp -r work_dirs/oake/v3det/clip_objects_cuda_train/output/* work_dirs/oake/v3det/clip_objects_cuda/output
cp -r work_dirs/oake/v3det/clip_objects_cuda_val/output/* work_dirs/oake/v3det/clip_objects_cuda/output
```

The number of files generated by OAKE-objects may be less than the number of images in the dataset.
Images without objects are skipped.

```bash
auto_torchrun tools/generate_sample_images.py lvis
python tools/encode_sample_images.py lvis
python tools/sample_visual_category_embeddings.py lvis clip
```

## DP

To conduct training for coco

```bash
auto_torchrun -m oadp.dp.train ov_coco configs/dp/ov_coco.py [--override .validator.dataloader.dataset.ann_file::data/coco/annotations/instances_val2017.48.json]
```

To conduct training for lvis

```bash
auto_torchrun -m oadp.dp.train ov_lvis configs/dp/ov_lvis.py  # --load-model-from pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth
```

To conduct training for objects365

```bash
auto_torchrun -m oadp.dp.train ov_objects365 configs/dp/ov_objects365.py
```

To conduct training for v3det

```bash
auto_torchrun -m oadp.dp.train ov_v3det configs/dp/ov_v3det.py
```

To test a specific checkpoint

```bash
auto_torchrun -m oadp.dp.test ov_coco configs/dp/ov_coco.py --load-model-from work_dirs/ov_coco/epoch_24.pth --visual xxx
auto_torchrun -m oadp.dp.test ov_lvis configs/dp/ov_lvis.py --load-model-from work_dirs/ov_lvis/epoch_24.pth --visual xxx
```

For the instance segmentation performance on LVIS, use the `metrics` argument

```bash
[DRY_RUN=True] auto_torchrun -m oadp.dp.test configs/dp/oadp_ov_lvis.py work_dirs/oadp_ov_lvis/epoch_24.pth --metrics bbox segm
```

NNI is supported but unnecessary.

```bash
DUMP=work_dirs/dump auto_torchrun -m oadp.dp.test configs/dp/oadp_ov_coco.py work_dirs/oadp_ov_coco/iter_32000.pth
DUMP=work_dirs/dump python tools/nni_dp_test.py
```

[Baidu Netdisk]: https://pan.baidu.com/s/1HXWYSN9Vk6yDhjRI19JrfQ?pwd=OADP

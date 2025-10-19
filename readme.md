# Distillation Pyramid for Multimodal Open-Vocabulary Object Detection

## Installation

### Prerequisites
- Python 3.11
- CUDA 11.8
- PyTorch 2.1.0

### Environment Setup

1. **Create a conda environment**
```bash
conda create -n mmdet3 python=3.11 -y
conda activate mmdet3
```

2. **Install PyTorch**
```bash
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --index-url https://download.pytorch.org/whl/cu118
```

3. **Install MMDetection ecosystem using OpenMIM**
```bash
pip install -U openmim
mim install mmengine==0.10.5
mim install mmcv==2.1.0
mim install mmdet==3.3.0
```

4. **Install other dependencies**
```bash
pip install -r requirements.txt
```

## Preparation
### Datasets

The expected directory structure for datasets:

```
data/
├── coco
│   ├── annotations
│   ├── mdetr_annotations
│   ├── train2014
│   ├── train2017
│   └── val2017
├── flickr30k_entities
│   ├── flickr30k_images
│   └── flickr_train_vg7.jsonl
├── gqa
│   ├── gqa_train_vg7.jsonl
│   └── images
├── mmovod
│   ├── merged.json
│   ├── pseudo_list.pth
│   └── samples
├── objects365
│   ├── annotations
│   └── train
├── qwen
│   ├── annotations
│   └── features
├── retrival
│   └── object_detection.json
└── v3det
    ├── annotations
    └── images
```

### Pretrained Models

Download the pretrained MM-Grounding-DINO models:

```
mm_grounding_dino/
├── grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
├── grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
└── grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
```

You can download them using:
```bash
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth
wget https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth
```

### OADP Features
Before using the distillation technique, we need to extract features offline. You can use the code in the `main` branch for extraction. The specific command is:



## Training
### Hardware Requirements

All experiments were conducted on 8x NVIDIA RTX 4090 (24GB) GPUs.

### Text-based Model Training

Train the text-based distillation model using EVA-CLIP features:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH torchrun --nproc_per_node=8 tools/train.py \
    configs/ov_distill_shortest_edge.py \
    --work-dir work_dirs/ov_distill_0.025_0.25_0.025_eva-clip_shortest_edge \
    --cfg-options \
        model.obj_loss_weight=0.025 \
        model.block_loss_weight=0.25 \
        model.global_loss_weight=0.025 \
        load_from=mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    --resume \
    --launcher pytorch
```

### Image-based Model Training

Train the image-based distillation model using LLM-extracted features:

```bash
PYTHONPATH=$(pwd):$PYTHONPATH torchrun --nproc_per_node=8 tools/train.py \
    configs/fs_llm_features_distill.py \
    --work-dir work_dirs/fs_distill_0.03_0.8_0.8 \
    --cfg-options \
        model.w_distill=0.03 \
        model.w_global=0.8 \
        model.w_structure=0.8 \
        load_from=mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth \
    --resume \
    --launcher pytorch
```


_base_ = [
    'coco_instance.py',
]

data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadCLIPFeatures',
        images=dict(
            type='PthAccessLayer',
            data_root=data_root + 'embeddings',
        ),
        regions=dict(
            type='PthAccessLayer',
            data_root=data_root + 'mask_embeddings',
        ),
    ),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='MaskToTensor', num_classes=80),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['clip_bboxes']),
    dict(type='ToDataContainer', fields=[
        dict(key='clip_patches'),
        dict(key='clip_bboxes'),
    ]),
    dict(type='Collect', keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
        'gt_masks',
        'gt_masks_tensor',
        'clip_image',
        'clip_patches',
        'clip_bboxes',
    ]),
]
data = dict(train=dict(pipeline=train_pipeline))
evaluation = dict(interval=1, metric='bbox')

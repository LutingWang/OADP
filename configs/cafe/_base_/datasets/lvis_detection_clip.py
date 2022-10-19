_base_ = [
    'coco_detection.py',
]

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadCLIPFeatures',
        task_name='',
        images=dict(
            type='PthAccessLayer',
            data_root='data/coco/embeddings',
            with_patches=False,
        ),
        regions=dict(
            type='PthAccessLayer',
            data_root='data/coco/vild_embeddings',
        ),
        # captions=dict(
        #     type='PthAccessLayer',
        #     data_root='data/coco/caption_embeddings',
        # ),
    ),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['clip_bboxes']),
    dict(type='ToDataContainer', fields=[
        dict(key='clip_patches'),
        dict(key='clip_bboxes'),
    ]),
    dict(type='Collect', keys=[
        'img', 'gt_bboxes', 'gt_labels',
        'clip_image',
        'clip_patches',
        'clip_bboxes',
        # 'clip_captions',
    ]),
]
data = dict(
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_train.json',
            img_prefix='data/coco/',
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix='data/coco/',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json',
        img_prefix='data/coco/',
    ),
)

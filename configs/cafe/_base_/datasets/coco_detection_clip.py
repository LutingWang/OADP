_base_ = [
    'coco_detection.py',
]

data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadCLIPFeatures',
        task_name='train',
        images=dict(
            type='PthAccessLayer',
            data_root=data_root + 'embeddings',
        ),
        regions=dict(
            type='PthAccessLayer',
            data_root=data_root + 'vild_embeddings',
            # data_root=data_root + 'mask_embeddings',
        ),
    ),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(
        type='Resize',
        img_scale=[(1330, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True,
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=[
        'clip_patches',
        'clip_patch_labels',
        'clip_bboxes',
    ]),
    dict(type='ToDataContainer', fields=[
        dict(key='clip_patch_feats'),
        dict(key='clip_patches'),
        dict(key='clip_patch_labels'),
        dict(key='clip_bbox_feats'),
        dict(key='clip_bboxes'),
    ]),
    dict(type='Collect', keys=[
        'img',
        'gt_bboxes',
        'gt_labels',
        'clip_image',
        'clip_patch_feats',
        'clip_patches',
        'clip_patch_labels',
        'clip_bbox_feats',
        'clip_bboxes',
        # 'clip_captions',
    ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadCLIPFeatures',
        task_name='val',
        images=dict(
            type='PthAccessLayer',
            data_root=data_root + 'embeddings',
        ),
        # regions=dict(
        #     type='PthAccessLayer',
        #     data_root=data_root + 'mask_embeddings',
        #     as_proposals=True,
        # ),
    ),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='ToTensor', keys=['clip_patches', 'proposals']),
            # dict(type='ToDataContainer', fields=[dict(key='proposals', stack=False)]),
            # dict(type='Collect', keys=['img', 'clip_patches', 'proposals']),
            dict(type='ToTensor', keys=['clip_patches']),
            dict(type='Collect', keys=['img', 'clip_patches']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

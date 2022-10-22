_base_ = [
    'coco_instance.py',
]

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadProposals'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadDetproFeatures',
        task_name='train2017',
        regions=dict(
            type='PthAccessLayer',
            data_root='data/lvis_v1/data/lvis_clip_image_embedding/',
        ),
    ),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='ToTensor', keys=['clip_bboxes']),
    dict(type='ToDataContainer', fields=[
        dict(key='clip_bbox_feats'),
        dict(key='clip_bboxes'),
    ]),
    dict(type='Collect', keys=[
        'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
        'clip_bbox_feats',
        'clip_bboxes',
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
            proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_train.pkl',
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

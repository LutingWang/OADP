_base_ = [
    'coco_detection_clip.py',
]

data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadCLIPFeatures',
        task_name='train',
        images=dict(
            type='PthAccessLayer',
            data_root=data_root + 'embeddings',
        ),
        regions=dict(
            type='PthAccessLayer',
            data_root=data_root + 'proposal_embeddings8',
        ),
    ),
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
        'gt_masks',
        'clip_image',
        'clip_patch_feats',
        'clip_patches',
        'clip_patch_labels',
        'clip_bbox_feats',
        'clip_bboxes',
        # 'clip_captions',
    ]),
]
data = dict(train=dict(pipeline=train_pipeline))
# evaluation = dict(interval=1, metric='bbox')

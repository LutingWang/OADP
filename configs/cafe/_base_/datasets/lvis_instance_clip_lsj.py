_base_ = [
    'coco_instance.py',
]

image_size = (512, 512)

dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadProposals'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(
    #     type='LoadDetproFeatures',
    #     images=dict(
    #         type='PthAccessLayer',
    #         data_root='data/coco/embeddings',
    #         task_name='',
    #     ),
    #     regions=dict(
    #         type='PthAccessLayer',
    #         data_root='data/lvis_v1/data/lvis_clip_image_embedding/',
    #         task_name='train2017',
    #     ),
    # ),
    dict(
        type='LoadCLIPFeatures4LVIS',
        task_name='',
        images=dict(
            type='PthAccessLayer',
            data_root='data/coco/embeddings',
        ),
        regions=dict(
            type='PthAccessLayer',
            # data_root='data/coco/vild_embeddings',
            data_root='data/coco/mask_embeddings',
        ),
    ),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),
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
            # proposal_file=data_root + 'proposals/rpn_r101_fpn_lvis_v1_train.pkl',
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

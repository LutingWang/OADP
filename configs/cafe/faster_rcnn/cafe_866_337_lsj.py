model = dict(
    type='Cafe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1203,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
                split='LVIS',
                num_base_classes=866,
                scaler=dict(train=0.01, val=0.007))),
        image_head=dict(with_reg=False),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            class_agnostic=True,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
            num_classes=1203),
        patch_head=dict(
            type='Shared2FCBBoxHead',
            with_reg=False,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
                split='LVIS',
                num_base_classes=866,
                scaler=dict(train=0.01, val=0.007)),
            loss=dict(
                type='AsymmetricLoss',
                weight=dict(type='WarmupScheduler', value=16, iter_=1000),
                gamma_neg=4,
                gamma_pos=0,
                clip=0.05,
                disable_torch_grad_focal_loss=True))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False,
            mask_size=28)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=300,
            mask_thr_binary=0.5)),
    distiller=dict(
        adapts=dict(),
        losses=dict(
            loss_clip_bboxes=dict(
                type='L1Loss',
                norm=True,
                fields=['bbox_feats', 'clip_bbox_feats'],
                weight=dict(type='WarmupScheduler', value=256, iter_=200)),
            loss_clip_bboxes_relation=dict(
                type='RKDLoss',
                fields=['bbox_feats', 'clip_bbox_feats'],
                weight=dict(type='WarmupScheduler', value=8, iter_=200)),
            loss_clip_patches=dict(
                type='L1Loss',
                norm=True,
                fields=['patch_feats', 'clip_patch_feats'],
                weight=dict(type='WarmupScheduler', value=128, iter_=200)),
            loss_clip_patches_relation=dict(
                type='RKDLoss',
                fields=['patch_feats', 'clip_patch_feats'],
                weight=dict(type='WarmupScheduler', value=8, iter_=200))),
        student_hooks=dict()),
    num_classes=1203)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=2.5e-05)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=16)
dataset_type = 'LVISV1Dataset866337'
data_root = 'data/lvis_v1/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='LoadCLIPFeatures4LVIS',
        task_name='',
        images=dict(type='PthAccessLayer', data_root='data/coco/embeddings'),
        regions=dict(
            type='PthAccessLayer', data_root='data/coco/mask_embeddings')),
    dict(
        type='Resize',
        img_scale=(1024, 1024),
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(1024, 1024),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='ToTensor',
        keys=['clip_patches', 'clip_patch_labels', 'clip_bboxes']),
    dict(
        type='ToDataContainer',
        fields=[
            dict(key='clip_patch_feats'),
            dict(key='clip_patches'),
            dict(key='clip_patch_labels'),
            dict(key='clip_bbox_feats'),
            dict(key='clip_bboxes')
        ]),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'clip_image',
            'clip_patch_feats', 'clip_patches', 'clip_patch_labels',
            'clip_bbox_feats', 'clip_bboxes'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=0.001,
        dataset=dict(
            type='LVISV1Dataset866337',
            ann_file='data/lvis_v1/annotations/lvis_v1_train.json.LVIS.866',
            img_prefix='data/coco/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='LoadCLIPFeatures4LVIS',
                    task_name='',
                    images=dict(
                        type='PthAccessLayer',
                        data_root='data/coco/embeddings'),
                    regions=dict(
                        type='PthAccessLayer',
                        data_root='data/coco/mask_embeddings')),
                dict(
                    type='Resize',
                    img_scale=(1024, 1024),
                    ratio_range=(0.1, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(1024, 1024),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size=(1024, 1024)),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='ToTensor',
                    keys=['clip_patches', 'clip_patch_labels', 'clip_bboxes']),
                dict(
                    type='ToDataContainer',
                    fields=[
                        dict(key='clip_patch_feats'),
                        dict(key='clip_patches'),
                        dict(key='clip_patch_labels'),
                        dict(key='clip_bbox_feats'),
                        dict(key='clip_bboxes')
                    ]),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                        'clip_image', 'clip_patch_feats', 'clip_patches',
                        'clip_patch_labels', 'clip_bbox_feats', 'clip_bboxes'
                    ])
            ])),
    val=dict(
        type='LVISV1Dataset866337',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json.LVIS',
        img_prefix='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='LVISV1Dataset866337',
        ann_file='data/lvis_v1/annotations/lvis_v1_val.json.LVIS',
        img_prefix='data/coco/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(
    interval=4, metric='bbox', classwise=True, tmpdir='work_dirs/tmp')
image_size = (1024, 1024)
fp16 = dict(loss_scale=dict(init_scale=64.0))
work_dir = 'work_dirs/lvis_lsj'
auto_resume = False
gpu_ids = range(0, 8)

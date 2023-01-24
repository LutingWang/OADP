_base_ = [
    'datasets/ov_coco.py',
    'models/oadp_faster_rcnn_r50_fpn.py',
    'schedules/schedule_40k.py',
]

model = dict(
    image_head=dict(
        topk=20,
        classifier=dict(
            type='Classifier',
            prompts='data/prompts/ml_coco.pth',
            in_features=256,
            out_features=65,
        ),
        loss=dict(
            type='AsymmetricLoss',
            weight=dict(type='WarmupScheduler', gain=4, end=2000),
            gamma_neg=4,
            gamma_pos=0,
        ),
    ),
    roi_head=dict(
        type='OADPRoIHead',
        bbox_head=dict(
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
        object_head=dict(
            cls_predictor_cfg=dict(
                # type='ViLDClassifier',
                # prompts='data/prompts/vild.pth',
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
        block_head=dict(
            type='BlockBBoxHead',
            topk=5,
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
            loss=dict(
                type='AsymmetricLoss',
                weight=dict(type='WarmupScheduler', gain=16, end=1000),
                gamma_neg=4,
                gamma_pos=0,
            ),
        ),
    ),
    distiller=dict(
        student_hooks=dict(
            image=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='._image_head._classifier._linear',
                ),
            ),
            blocks=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='.roi_head._block_head.fc_cls._linear',
                ),
            ),
            objects=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='.roi_head._object_head.fc_cls._linear',
                ),
            ),
        ),
        adapts=dict(),
        losses=dict(
            loss_clip_objects=dict(
                inputs=('objects', 'clip_objects'),
                action=dict(
                    type='L1Loss',
                    weight=dict(type='WarmupScheduler', gain=256, end=200),
                ),
            ),
            loss_clip_image=dict(
                inputs=('image', 'clip_image'),
                action=dict(
                    type='MSELoss',
                    weight=dict(type='WarmupScheduler', gain=0.5, end=200),
                    reduction='sum',
                ),
            ),
            loss_clip_blocks=dict(
                inputs=('blocks', 'clip_blocks'),
                action=dict(
                    type='L1Loss',
                    weight=dict(type='WarmupScheduler', gain=128, end=200),
                ),
            ),
            loss_clip_block_relations=dict(
                inputs=('blocks', 'clip_blocks'),
                action=dict(
                    type='RKDLoss',
                    weight=dict(type='WarmupScheduler', gain=8, end=200),
                ),
            ),
        ),
    ),
)
trainer = dict(
    load_from='pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth',
    optimizer=dict(
        weight_decay=2.5e-5,
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
)

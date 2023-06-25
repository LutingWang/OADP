_base_ = [
    'vild_ensemble_faster_rcnn_r50_fpn.py',
]

model = dict(
    type='OADP',
    backbone=dict(
        style='caffe',
        init_cfg=None,
    ),
    global_head=dict(
        topk=20,
        classifier=dict(
            type='Classifier',
            prompts='data/prompts/ml_coco.pth',
            in_features=256,
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
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
    ),
    distiller=dict(
        student_hooks=dict(
            global_=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='._global_head._classifier._linear',
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
            loss_clip_global=dict(
                inputs=('global_', 'clip_global'),
                action=dict(
                    type='MSELoss',
                    weight=dict(type='WarmupScheduler', gain=0.5, end=200),
                    # FIXME: in todd, sum should multiply by number of GPUs
                    reduction='sum',
                ),
            ),
        ),
    ),
    test_cfg=dict(rcnn=dict(
        score_thr=0.0,
        max_per_img=300,
    )),
)

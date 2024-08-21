_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    type='ViLD',
    backbone=dict(style='caffe', init_cfg=None),
    neck=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_classes=None,
            reg_class_agnostic=True,
        ),
        object_head=dict(type='Shared4Conv1FCObjectBBoxHead'),
    ),
    distiller=dict(
        type='SelfDistiller',
        student_hook_pipeline=dict(
            objects=dict(
                type='SingleOperator',
                args=tuple(),
                atom=dict(
                    type=(
                        'TaskRegistry.KDRegistry.KDDistillerRegistry.'
                        'KDHookRegistry.Hook'
                    ),
                    path='.roi_head._object_head.fc_cls._linear',
                ),
            ),
        ),
        adapt_pipeline=dict(),
        loss_pipeline=dict(
            loss_clip_objects=dict(
                type='SingleOperator',
                args=('objects', 'clip_objects'),
                atom=dict(
                    type='ModelRegistry.LossRegistry.L1Loss',
                    weight=dict(type='WarmupScheduler', gain=256, end=200),
                ),
            ),
        ),
    ),
    test_cfg=dict(rcnn=dict(
        score_thr=0.0,
        max_per_img=300,
    )),
)

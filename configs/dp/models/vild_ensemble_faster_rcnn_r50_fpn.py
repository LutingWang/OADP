_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    neck=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            num_classes=None,
        ),
        object_head=dict(type='Shared4Conv1FCObjectBBoxHead'),
    ),
)

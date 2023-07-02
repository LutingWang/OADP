_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
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
    test_cfg=dict(rcnn=dict(
        score_thr=0.0,
        max_per_img=300,
    )),
)

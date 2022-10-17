_base_ = [
    '../_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',

    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_866_337.py',

    '../_base_/datasets/lvis_detection_clip.py',
    '../_base_/datasets/lvis_866_337.py',

    'mixins/dcp.py',
]

model = dict(
    type='Cafe',
    backbone=dict(
        style='caffe',
        # frozen_stages=4,
        init_cfg=None,
    ),
    neck=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    roi_head=dict(
        bbox_head=dict(
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            # norm_cfg=None,
            num_classes=1203,
        ),
    ),
    distiller=dict(
        student_hooks=dict(),
        adapts=dict(),
        losses=dict(),
    ),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.0,
            max_per_img=300,
        ),
    ),
)
load_from = 'data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'
optimizer = dict(
    weight_decay=2.5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'neck': dict(lr_mult=0.1),
            'roi_head.bbox_head': dict(lr_mult=0.5),
        },
    ),
)

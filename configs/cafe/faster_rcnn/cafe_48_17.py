_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/models/dh_faster_rcnn_r50_fpn.py',
    '../_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',

    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_48_17.py',

    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/coco_48_17.py',

    # dcp
    '../_base_/datasets/coco_detection_clip_48_17.py',
    # '../_base_/datasets/coco_48_17.py',
    'mixins/dcp.py',

]

model = dict(
    type='Cafe',
    backbone=dict(
        # frozen_stages=3,
        style='caffe',
        init_cfg=None,
    ),
    neck=dict(
        # norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    roi_head=dict(
        bbox_head=dict(
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
        ),
    ),
    distiller=dict(
        student_hooks=dict(),
        adapts=dict(),
        losses=dict(),
    ),
)
load_from='data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'
optimizer = dict(weight_decay=2.5e-5)

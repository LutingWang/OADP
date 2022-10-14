_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/models/dh_faster_rcnn_r50_fpn.py',
    '../_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',

    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_48_17.py',

    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/coco_48_17.py',

    '../_base_/datasets/coco_instance_clip.py',
    '../_base_/datasets/coco_48_17.py',
    'mixins/mask.py',
    # 'mixins/post.py',
    'mixins/dcp.py',

]

model = dict(
    type='Cafe',
    backbone=dict(
        style='caffe',
        init_cfg=None,
    ),
    neck=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    roi_head=dict(
        bbox_head=dict(
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            # norm_cfg=None,
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
optimizer = dict(weight_decay=2.5e-5)
runner = dict(max_epochs=6)
lr_config = dict(step=[4])

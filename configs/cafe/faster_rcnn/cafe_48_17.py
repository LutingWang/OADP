_base_ = [
    # '../_base_/models/faster_rcnn_r50_fpn.py',
    # '../_base_/models/dh_faster_rcnn_r50_fpn.py',
    '../_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',

    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_48_17.py',

    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/coco_48_17.py',

    '../_base_/datasets/coco_detection_clip.py',
    '../_base_/datasets/coco_48_17.py',
    'mixins/dcp.py',
    # 'mixins/multilabel_48_17.py',
    # 'mixins/post.py',

    'mixins/patch_48_17.py',

    # 'mixins/ext.py',
    'mixins/coco_48_17_ext.py',
]

model = dict(
    type='Cafe',
    num_classes=65,
    backbone=dict(
        style='caffe',
        # frozen_stages=4,
        init_cfg=None,
    ),
    neck=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    # rpn_head=dict(
    #     loss_cls=dict(loss_weight=0.1),
    #     loss_bbox=dict(loss_weight=0.1),
    # ),
    roi_head=dict(
        bbox_head=dict(
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            # norm_cfg=None,
            # loss_cls=dict(loss_weight=0.1),
            # loss_bbox=dict(loss_weight=0.1),
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

# runner = dict(max_epochs=6)
# lr_config = dict(step=[4])
runner = dict(
    _delete_=True,
    type='IterBasedRunner',
    max_iters=60000,
)
lr_config = dict(
    by_epoch=False,
    step=[30000, 50000],
)
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ],
)
interval = 2000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False,
    interval=interval * 2,
    save_last=True,
)
evaluation = dict(
    interval=interval,
)

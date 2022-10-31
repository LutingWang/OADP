_base_ = [
    '../_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',

    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_866_337.py',

    '../_base_/datasets/lvis_instance_clip.py',
    '../_base_/datasets/lvis_866_337.py',

    'mixins/dcp.py',
    'mixins/mask_866_337.py',

    'mixins/multilabel_866_337.py',
    'mixins/post.py',
]

model = dict(
    type='Cafe',
    num_classes=1203,
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
fp16 = dict(loss_scale=dict(init_scale=64.0))

# load_from = 'data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth'
# evaluation = dict(interval=4)
# optimizer = dict(
#     weight_decay=2.5e-5,
# )

load_from = 'work_dirs/lvis_debug_2/epoch_20.pth'
evaluation = dict(interval=1)
optimizer = dict(
    lr=0.002,
    weight_decay=2.5e-5,
)
lr_config = dict(
    step=[6],
)
runner = dict(max_epochs=8)
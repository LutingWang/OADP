_base_ = [
    'datasets/ov_coco.py',
    'models/oadp_faster_rcnn_r50_fpn.py',
    'schedules/40k.py',
    'schedules/oadp.py',
    'runtime.py',
]

model = dict(global_head=dict(classifier=dict(out_features=65)))
trainer = dict(
    optimizer=dict(
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
)

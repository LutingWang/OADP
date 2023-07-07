_base_ = [
    'datasets/ov_coco.py',
    'models/vild_ensemble_faster_rcnn_r50_fpn.py',
    'models/oadp.py',
    'models/mask.py',
    'schedules/40k.py',
    'schedules/oadp.py',
    'runtime.py',
]

model=dict(
    roi_head=dict(
        object_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
    ),
)

trainer = dict(
    optimizer=dict(
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
)
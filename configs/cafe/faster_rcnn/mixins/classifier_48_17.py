_base_ = [
    'classifier.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                split='COCO_48_17',
            ),
        ),
    ),
)

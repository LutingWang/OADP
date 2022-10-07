_base_ = [
    'classifier.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=65,
            reg_class_agnostic=False,
            cls_predictor_cfg=dict(
                split='COCO_48_17',
                num_base_classes=48,
            ),
        ),
    ),
)

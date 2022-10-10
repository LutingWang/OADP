_base_ = [
    'classifier.py',
]

model = dict(
    cls_predictor_cfg=dict(
        type='ViLDClassifier',
        pretrained='data/coco/prompt/vild_coco.pth',
        split='COCO_48_17',
        num_base_classes=48,
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=65,
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/coco/prompt/vild_coco.pth',
                split='COCO_48_17',
                num_base_classes=48,
            ),
        ),
    ),
)

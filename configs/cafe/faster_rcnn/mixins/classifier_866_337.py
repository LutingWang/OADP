_base_ = [
    'classifier.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/coco/prompt/vild_lvis.pth',
                split='LVIS',
                num_base_classes=866,
            ),
        ),
    ),
)

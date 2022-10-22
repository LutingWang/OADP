_base_ = [
    'classifier.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                # pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
                pretrained='data/lvis_v1/prompt/vild_lvis.pth',
                split='LVIS',
                num_base_classes=866,
            ),
        ),
    ),
)

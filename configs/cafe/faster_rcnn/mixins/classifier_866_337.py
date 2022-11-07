_base_ = [
    'classifier.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=1203,
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
                # pretrained='data/prompts/vild.pth',
                split='LVIS',
                num_base_classes=866,
                scaler=dict(  # this is same with vild
                    train=0.01,
                    val=0.007,
                ),
            ),
        ),
        aux_image_head=dict(
            cls_predictor_cfg=dict(
                pretrained='data/prompts/vild.pth',
            ),
        ),
    ),
)

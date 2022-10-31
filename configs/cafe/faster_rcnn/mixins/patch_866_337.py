_base_ = [
    'patch.py',
]

model = dict(
    roi_head=dict(
        patch_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
                split='LVIS',
                num_base_classes=866,
                scaler=dict(  # this is same with detpro
                    train=0.01,
                    val=0.007,
                ),
            ),
        ),
    ),
)

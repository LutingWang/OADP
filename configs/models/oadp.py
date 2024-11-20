_base_ = ['./faster-rcnn_r50_fpn.py']

cls_predictor_cfg = dict(
    type='OADPClassifier',
    text_model=dict(
        type="HuggingCLIPLanguageBackbone",
        model_name='openai/clip-vit-base-patch32',
    )
)

model = dict(
    type='OADP',
    roi_head=dict(
        bbox_head=dict(cls_predictor_cfg=cls_predictor_cfg),
    ),
)
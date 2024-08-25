_base_ = [
    'ov_lvis.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(_delete_=True, type='FewShotClassifier'),
        ),
    ),
    visual_embedding=True,
)

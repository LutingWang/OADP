_base_ = [
    'oadp_ov_coco.py',
]

model = dict(
    roi_head=dict(
        object_head=dict(
            cls_predictor_cfg=dict(_delete_=True, type='FewShotClassifier'),
        ),
    ),
    visual_embedding=True,
)

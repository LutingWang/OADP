_base_ = [
    'datasets/ov_coco.py',
    'models/oadp_faster_rcnn_r50_fpn.py',
    'schedules/40k.py',
    'base.py',
]

model = dict(
    global_head=dict(
        classifier=dict(
            type='Classifier',
            prompts='data/prompts/ml_coco.pth',
            out_features=65,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(cls_predictor_cfg=dict(type='FewShotClassifier')),
        object_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
    ),
    visual_embedding=dict(
        type='VisualEmbedding',
        loader=dict(type='COCOLoader'),
    )
)

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'roi_head.bbox_head': dict(lr_mult=0.5)})
)

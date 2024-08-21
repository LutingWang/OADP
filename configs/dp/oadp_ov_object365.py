_base_ = [
    'datasets/objects365.py',
    'models/oadp_faster_rcnn_r50_fpn.py',
    'schedules/40k.py',
    'base.py',
]

model = dict(
    global_head=dict(
        classifier=dict(
            type='ViLDClassifier',
            prompts='data/prompts/vild.pth',
            out_features=365,
        ),
    ),
    roi_head=dict(
        # bbox_head=dict(
        #     cls_predictor_cfg=dict(
        #         type='ViLDClassifier',
        #         prompts='data/prompts/vild.pth',
        #     ),
        # ),
        bbox_head=dict(cls_predictor_cfg=dict(type='FewShotClassifier')),
        object_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
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

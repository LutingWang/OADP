_base_ = [
    'oadp_ov_coco.py'
]

model = dict(
    global_head=dict(
        classifier=dict(
            type='LLMClassifier',
            prompts='data/prompts/llm.pth',
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='LLMClassifier',
                prompts='data/prompts/llm.pth',
            ),
        ),
        object_head=dict(
            cls_predictor_cfg=dict(
                type='LLMClassifier',
                prompts='data/prompts/llm.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                type='LLMClassifier',
                prompts='data/prompts/llm.pth',
            ),
        ),
    ),
)

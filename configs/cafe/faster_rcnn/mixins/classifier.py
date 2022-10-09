model = dict(
    # compat double heads
    cls_predictor_cfg=dict(
        type='Classifier',
        pretrained='data/coco/prompt/prompt1.pth',
        split='COCO',
    ),
    roi_head=dict(
        bbox_head=dict(
            # compat vild ensemble head
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='data/coco/prompt/prompt1.pth',
                split='COCO',
            ),
        ),
    ),
)

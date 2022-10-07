model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='data/coco/prompt/prompt1.pth',
                split='COCO',
            ),
        ),
    ),
)

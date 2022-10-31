model = dict(
    roi_head=dict(
        patch_head=dict(
            cls_predictor_cfg=dict(
                pretrained='data/coco/prompt/prompt1.pth',
                split='COCO_48_17',
                num_base_classes=48,
            ),
        ),
    ),
)

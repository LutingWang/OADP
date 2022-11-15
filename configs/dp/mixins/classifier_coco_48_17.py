model = dict(
    roi_head=dict(
        bbox_head=dict(
            reg_class_agnostic=True,
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                pretrained='data/prompts/vild.pth',
                split='COCO_48_17',
            ),
        ),
        image_head=dict(
            cls_predictor_cfg=dict(
                # type='ViLDClassifier',
                # pretrained='data/prompts/vild.pth',
                type='Classifier',
                pretrained='data/prompts/ml_coco.pth',
                split='COCO_48_17',
            ),
        ),
    ),
)

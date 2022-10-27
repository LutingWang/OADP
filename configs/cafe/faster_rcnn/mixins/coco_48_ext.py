data = dict(
    train=dict(
        type='CocoDataset48Ext',
        ann_file='data/coco/annotations/instances_train2017.json.COCO_48_EXT',
    ),
)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDExtClassifier',
                pretrained='data/prompts/vild.pth',
                split=dict(
                    train='data/coco/annotations/instances_train2017.json.COCO_48_EXT',
                    val='COCO_48_17',
                ),
                num_base_classes=None
            ),
        ),
    ),
)

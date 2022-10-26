data = dict(
    train=dict(
        ann_file='data/coco/annotations/instances_train2017.json.COCO_48_17.ext',
    ),
)
model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                num_base_classes=None,
            ),
        ),
    ),
)

dataset_type = 'CocoDataset4817'
data_root = 'data/coco/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json.COCO_48_17.48',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json.COCO_48_17.filtered',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json.COCO_48_17.filtered',
    ),
)
evaluation = dict(classwise=True)

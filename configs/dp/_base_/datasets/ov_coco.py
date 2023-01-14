categories = 'coco'
dataset_type = 'OV_COCO'
ann_file_prefix = 'data/coco/annotations/instances_'
trainer = dict(
    dataloader=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'train2017.48.json',
        ),
    ),
)
validator = dict(
    dataloader=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file_prefix + 'val2017.65.min.json',
        ),
    ),
)

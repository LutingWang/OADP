_base_ = [
    'base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/images/train2017'),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/coco/oake/images/val2017'),
    ),
)

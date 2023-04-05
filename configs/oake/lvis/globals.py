_base_ = [
    '../base.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/lvis_v1/oake/globals/train2017'),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(output_dir='data/lvis_v1/oake/globals/val2017'),
    ),
)

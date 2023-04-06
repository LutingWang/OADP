_base_ = [
    'objects.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            output_dir='data/lvis_v1/oake/objects/train2017',
            proposal_file='data/lvis_v1/proposals/debug_train.pkl',
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            output_dir='data/lvis_v1/oake/objects/val2017',
            proposal_file='data/lvis_v1/proposals/debug_val.pkl',
        ),
    ),
)

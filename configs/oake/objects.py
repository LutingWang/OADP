_base_ = [
    '../strategies/cuda.py',
    'base.py',
]

runner_type = 'ObjectValidator'
dataset_type = 'ObjectDataset'
mini_batch_size = 512
trainer = dict(
    type=runner_type,
    _override_={
        '.callbacks[0].interval': 5,
    },
    dataset=dict(
        type=dataset_type,
        proposal_file='data/coco/proposals/oln_r50_fpn_coco_train.pkl',
        proposal_sorted=True,
    ),
    mini_batch_size=mini_batch_size,
)
validator = dict(
    type=runner_type,
    _override_={
        '.callbacks[0].interval': 5,
    },
    dataset=dict(
        type=dataset_type,
        proposal_file='data/coco/proposals/oln_r50_fpn_coco_val.pkl',
        proposal_sorted=True,
    ),
    mini_batch_size=mini_batch_size,
)

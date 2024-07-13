_base_ = [
    '../strategies/ddp.py',
    'base.py',
]

runner_type = 'BlockValidator'
dataset_type = 'BlockDataset'
trainer = dict(type=runner_type, dataset=dict(type=dataset_type))
validator = dict(type=runner_type, dataset=dict(type=dataset_type))

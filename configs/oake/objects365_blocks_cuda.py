_base_ = [
    'blocks_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365V2BlockDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365V2BlockDataset', split='val'),
)

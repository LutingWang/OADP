_base_ = [
    'blocks_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOBlockDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOBlockDataset', split='val'),
)

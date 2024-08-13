_base_ = [
    'objects_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.LVISObjectDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.LVISObjectDataset', split='val'),
)

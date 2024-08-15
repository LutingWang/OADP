_base_ = [
    'objects_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365ObjectDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365ObjectDataset', split='val'),
)

_base_ = [
    'objects_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOObjectDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOObjectDataset', split='val'),
)

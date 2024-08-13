_base_ = [
    'globals_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOGlobalDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.COCOGlobalDataset', split='val'),
)

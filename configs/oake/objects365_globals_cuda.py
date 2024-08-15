_base_ = [
    'globals_cuda.py',
]

trainer = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365GlobalDataset', split='train'),
)
validator = dict(
    dataset=dict(type='OAKEDatasetRegistry.Objects365GlobalDataset', split='val'),
)

_base_ = [
    'objects.py',
]

dataset_type = 'LVISObjectDataset'
trainer = dict(
    dataset=dict(
        type=dataset_type,
        annotations_file='data/lvis_v1/annotations/lvis_v1_train.json',
        proposal_file='data/lvis_v1/proposals/oln_r50_fpn_lvis_train.pkl',
    ),
)
validator = dict(
    dataset=dict(
        type=dataset_type,
        annotations_file='data/lvis_v1/annotations/lvis_v1_val.json',
        proposal_file='data/lvis_v1/proposals/oln_r50_fpn_lvis_val.pkl',
    ),
)

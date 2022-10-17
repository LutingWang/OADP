dataset_type = 'LVISV1Dataset866337'
data_root = 'data/lvis_v1/'
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_train.json.LVIS.866',
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json.LVIS',
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/lvis_v1_val.json.LVIS',
    ),
)
evaluation = dict(classwise=True)

model = dict(
    type='defult',
)

val_cfg=dict()

val_evaluator = [
    dict(
        type='MutiLabelMetric',
        threshold=0.1,
    ),
]

val_dataset = dict(
    type='LVISDataset',
    data_root='data/lvis',
    ann_file='annotations/lvis_v1_val.json',
    pipeline=[
        dict(type='LoadImage'),
        dict(type='CLIPTransforms'),
        dict(type='PackData'),
    ],
)

val_dataloader = dict(
    batch_size=32,
    dataset=val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate')
)

work_dir = 'work_dirs/'
launcher = 'pytorch'
log_processor = dict(window_size=1)
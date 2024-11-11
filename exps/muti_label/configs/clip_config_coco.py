_base_ = ['base.py']

val_dataset = dict(
    type='COCODatasets',
    data_root='data/coco2017',
    data_prefix=dict(
        img_path='val2017',
    ),
    ann_file='annotations/instances_val2017.json',
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

model = dict(
    type='CLIPModel',
)
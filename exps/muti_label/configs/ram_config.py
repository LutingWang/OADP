_base_ = ['base.py']

val_dataset = dict(
    type='LVISDataset',
    data_root='data/lvis',
    ann_file='annotations/lvis_v1_minival.1203.json',
    pipeline=[
        dict(type='LoadImage'),
        dict(type='RAMTransforms'),
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
    type='RAMModel',
    model_path='pretrained/ram/ram_plus_swin_large_14m.pth',
    llm_tag_des='data/lvis/annotations/openset_label_embedding.pth'
)

val_evaluator = [
    dict(
        type='MutiLabelMetric',
        threshold=0.5,
    ),
]
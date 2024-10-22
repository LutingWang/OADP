categories = 'v3det'
dataset_type = 'V3DetDataset'
data_root = 'data/v3det/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='V3DetTransform'),
    dict(
        type='MMLoadOAKEGlobal',
        access_layer=dict(type='V3DetGlobalAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEBlock',
        access_layer=dict(type='V3DetBlockAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEObject',
        access_layer=dict(type='V3DetObjectAccessLayer', model='clip'),
    ),
    dict(type='MMAssignOAKEBlockLabels'),
    dict(type='AppendBBoxes'),
    dict(
        type='RandomResize',
        scale=[(1330, 640), (1333, 800)],
        keep_ratio=True,
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackTrainInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackValInputs'),
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='annotations/v3det_2023_v1_train.json',
        filter_cfg=dict(
            filter_empty_gt=True,
            min_size=32,
            filter_oake=True,
            pipeline=train_pipeline,
            backend_args=None,
        ),
        pipeline=train_pipeline,
        backend_args=None,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img=''),
        ann_file='annotations/v3det_2023_v1_val.json',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
        filter_cfg=dict(
            filter_oake=True,
        ),
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/v3det_2023_v1_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator
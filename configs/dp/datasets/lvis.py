categories = 'lvis'
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LVISTransform'),
    dict(
        type='MMLoadOAKEGlobal',
        access_layer=dict(type='LVISGlobalAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEBlock',
        access_layer=dict(type='LVISBlockAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEObject',
        access_layer=dict(type='LVISObjectAccessLayer', model='clip'),
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
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='annotations/lvis_v1_train.866.json',
            data_prefix=dict(img=''),
            filter_cfg=dict(
                filter_empty_gt=True,
                min_size=32,
                filter_oake=True,
                pipeline=train_pipeline,
                backend_args=None,
            ),
        )
    )
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
        ann_file='annotations/lvis_v1_minival.1203.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='LVISMetric',
    ann_file=data_root + 'annotations/lvis_v1_minival.1203.json',
    metric=['bbox'],
    backend_args=None,
)
test_evaluator = val_evaluator

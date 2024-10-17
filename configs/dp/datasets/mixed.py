categories = 'v3det_objects365'


v3det_train_pipeline = [
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

v3det_dataset=dict(
    type='V3DetDataset',
    data_root='data/v3det/',
    data_prefix=dict(img=''),
    ann_file='annotations/v3det_2023_v1_train.json',
    filter_cfg=dict(
        filter_empty_gt=True,
        min_size=32,
        filter_oake=True,
        pipeline=v3det_train_pipeline,
        backend_args=None,
    ),
    pipeline=v3det_train_pipeline,
    backend_args=None,
)

objects365_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Objects365Transform'),
    dict(type='Objects365MixedV3DetTransform'),
    dict(
        type='MMLoadOAKEGlobal',
        access_layer=dict(type='Objects365GlobalAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEBlock',
        access_layer=dict(type='Objects365BlockAccessLayer', model='clip'),
    ),
    dict(
        type='MMLoadOAKEObject',
        access_layer=dict(type='Objects365ObjectAccessLayer', model='clip'),
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

objects365_dataset=dict(
    type='Objects365Dataset',
    data_root='data/objects365/',
    data_prefix=dict(img='train/'),
    ann_file='annotations/zhiyuan_objv2_train.json',
    filter_cfg=dict(
        filter_empty_gt=True,
        min_size=32,
        filter_oake=True,
        pipeline=objects365_train_pipeline,
        backend_args=None,
    ),
    pipeline=objects365_train_pipeline,
    backend_args=None,
)

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[v3det_dataset, objects365_dataset],
    ),
)

test_dataloader = val_dataloader = None
test_evaluator = val_evaluator = None
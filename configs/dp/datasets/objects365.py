categories = 'objects365'
dataset_type = 'Objects365V2Dataset'
data_root = 'data/objects365/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Objects365Transform'),
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
        data_prefix=dict(img='train/'),
        ann_file='annotations/zhiyuan_objv2_train.json',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
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
        data_prefix=dict(img='val/'),
        ann_file='annotations/zhiyuan_objv2_val.json',
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/zhiyuan_objv2_val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator

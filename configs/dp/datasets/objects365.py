_base_ = [
    'coco_detection.py',
]

dataset_type = 'Objects365V2Dataset'
categories = 'objects365'
data_root = 'data/object365v2/'
oake_root = 'work_dirs/oake/'
ann_file_prefix = 'annotations/zhiyuan_objv2_'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='LoadCLIPFeatures',
        default=dict(
            task_name='train2017',
            type='PthAccessLayer',
        ),
        globals_=dict(data_root=oake_root + 'globals'),
        blocks=dict(data_root=oake_root + 'blocks'),
        objects=dict(data_root=oake_root + 'objects/coco/output'),
    ),
    dict(
        type='RandomResize', scale=[(1330, 640), (1333, 800)], keep_ratio=True
    ),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackInputs',
        extra_keys=[
            'clip_global', 'clip_blocks', 'block_bboxes', 'block_labels',
            'clip_objects', 'object_bboxes'
        ],
    ),
]
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_file_prefix + 'train.json',
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='train/'),
        ann_file=ann_file_prefix + 'val.json',
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + ann_file_prefix + 'val.json',
    metric='bbox',
    format_only=False,
    backend_args=None
)
test_evaluator = val_evaluator

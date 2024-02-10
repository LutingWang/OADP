_base_ = [
    'coco_detection.py',
]

categories = 'coco'
data_root = 'data/coco/'
oake_root = data_root + 'oake/'
ann_file_prefix = 'annotations/instances_'

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
        objects=dict(data_root=oake_root + 'objects'),
    ),
    dict(type='RandomResize', scale=[(1330, 640), (1333, 800)], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackInputs',
        extra_keys=[
            'clip_global', 'clip_blocks', 'block_bboxes', 
            'block_labels', 'clip_objects', 'object_bboxes'
        ],
    ),
]
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_file_prefix + 'train2017.48.json',
        pipeline=train_pipeline,
    )
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        ann_file=ann_file_prefix + 'val2017.65.min.json',
    )    
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='OVCOCOMetric',
    ann_file=data_root + ann_file_prefix + 'val2017.65.min.json',
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
_base_ = [
    'lvis_v0.5_instance.py',
]

categories = 'lvis'
dataset_type = 'LVISV1Dataset'
data_root = 'data/lvis_v1/'
oake_root = data_root + 'oake/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            pipeline=train_pipeline,
            ann_file='annotations/lvis_v1_train.866.json',
            data_prefix=dict(img='')
        )
    )
)
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/lvis_v1_val.1203.json',
        data_prefix=dict(img='')
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/lvis_v1_val.1203.json')
test_evaluator = val_evaluator

_base_ = [
    'coco_instance.py',
]

dataset_type = 'OV_LVIS'
data_root = 'data/lvis_v1/'
norm = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
trainer = dict(
    dataloader=dict(
        dataset=dict(
            _delete_=True,
            type='ClassBalancedDataset',
            oversample_thr=1e-3,
            dataset=dict(
                type=dataset_type,
                ann_file=data_root + 'annotations/lvis_v1_train.json',
                img_prefix='data/coco/',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='LoadAnnotations', with_bbox=True, with_mask=True
                    ),
                    dict(
                        type='Resize',
                        img_scale=[(1333, 640), (1333, 800)],
                        multiscale_mode='range',
                        keep_ratio=True
                    ),
                    dict(type='RandomFlip', flip_ratio=0.5),
                    dict(type='Normalize', **norm),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(
                        type='Collect',
                        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']
                    ),
                ],
            ),
        ),
    ),
)
validator = dict(
    dataloader=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'annotations/lvis_v1_val.json',
            img_prefix='data/coco/',
        ),
    ),
)
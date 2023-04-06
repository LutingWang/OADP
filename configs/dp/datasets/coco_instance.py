_base_ = [
    'coco_detection.py',
]

norm = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)
trainer = dict(
    dataloader=dict(
        dataset=dict(
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
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
)

_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py',
    '../_base_/datasets/coco_48_17.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=65,
        ),
    ),
)

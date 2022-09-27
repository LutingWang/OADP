_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='CustomBBoxHead',
            cls_predictor_cfg=dict(
                type='Classifier',
            ),
        ),
    ),
)

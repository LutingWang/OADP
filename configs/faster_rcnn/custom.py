_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py',
]

model = dict(
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='work_dirs/prompt/epoch_3_classes.pth',
            ),
        ),
    ),
)

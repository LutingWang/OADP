_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py',
    'mixins/multilabel.py',
    'mixins/pre.py',
    'mixins/post.py',
    'mixins/awloss.py',
    'mixins/dci.py',
    'mixins/dcp.py',
]

model = dict(
    type='Cafe',
    distiller=dict(
        student_hooks=dict(),
        adapts=dict(),
        losses=dict(),
    ),
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='work_dirs/prompt/epoch_3_classes.pth',
            ),
        ),
    ),
)

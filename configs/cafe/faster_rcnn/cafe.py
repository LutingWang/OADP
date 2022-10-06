_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',

    # # multilabel
    # '../_base_/datasets/coco_detection.py',
    # 'mixins/multilabel.py',

    # # multilabel_pre
    # '../_base_/datasets/coco_detection.py',
    # 'mixins/multilabel.py',
    # 'mixins/pre.py',

    # # multilabel_post
    # '../_base_/datasets/coco_detection.py',
    # 'mixins/multilabel.py',
    # 'mixins/post.py',

    # # multilabel_post_awloss
    # '../_base_/datasets/coco_instance.py',
    # 'mixins/multilabel.py',
    # 'mixins/post.py',
    # 'mixins/awloss.py',

    # # multilabel_dcp
    # '../_base_/datasets/coco_detection_clip.py',
    # 'mixins/multilabel.py',
    # 'mixins/dcp.py',

    # # multilabel_dci
    # '../_base_/datasets/coco_detection_clip.py',
    # 'mixins/multilabel.py',
    # 'mixins/dci.py',

    # # multilabel_dcp_dci
    # '../_base_/datasets/coco_detection_clip.py',
    # 'mixins/multilabel.py',
    # 'mixins/dcp.py',
    # 'mixins/dci.py',

    # full
    '../_base_/datasets/coco_instance_clip.py',
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
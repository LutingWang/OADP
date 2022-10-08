_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
    'mixins/classifier_48_17.py',

    # '../_base_/datasets/coco_detection.py',
    # '../_base_/datasets/coco_48_17.py',

    # dcp
    '../_base_/datasets/coco_detection_clip.py',
    '../_base_/datasets/coco_48_17.py',
    'mixins/dcp.py',

]

model = dict(
    type='Cafe',
    distiller=dict(
        student_hooks=dict(),
        adapts=dict(),
        losses=dict(),
    ),
)

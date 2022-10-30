_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        type='ViLDRoIHead',
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
        ),
    ),
)

_base_ = [
    'faster_rcnn_r50_fpn.py',
]

model = dict(
    roi_head=dict(
        type='DoubleHeadRoIHead',
        reg_roi_scale_factor=1.3,
        bbox_head=dict(
            type='DoubleConvFCBBoxHead',
            num_convs=4,
            num_fcs=2,
            conv_out_channels=1024,
            loss_cls=dict(loss_weight=2.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=2.0),
        ),
    ),
)

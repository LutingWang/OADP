_base_ = ['./faster-rcnn_r50_fpn.py']


model = dict(
    type='OADP',
    roi_head=dict(bbox_head=dict(num_classes=365))
)
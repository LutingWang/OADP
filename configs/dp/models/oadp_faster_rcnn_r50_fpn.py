_base_ = [
    'vild_ensemble_faster_rcnn_r50_fpn.py',
]

model = dict(
    type='OADP',
    backbone=dict(
        style='caffe',
        init_cfg=None,
    ),
    test_cfg=dict(rcnn=dict(
        score_thr=0.0,
        max_per_img=300,
    )),
)

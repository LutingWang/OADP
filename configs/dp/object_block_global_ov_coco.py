_base_ = [
    'datasets/coco_detection.py',
    'datasets/ov_coco.py',
    'models/vild_ensemble_faster_rcnn_r50_fpn.py',
    'schedules/schedule_40k.py',
    'mixins/classifier_ov_coco.py',
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
trainer = dict(
    load_from='data/ckpts/soco_star_mask_rcnn_r50_fpn_400e.pth',
    optimizer=dict(
        weight_decay=2.5e-5,
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
)

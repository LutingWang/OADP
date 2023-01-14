_base_ = [
    '_base_/datasets/coco_detection.py',
    '_base_/datasets/ov_coco.py',
    '_base_/models/vild_ensemble_faster_rcnn_r50_fpn.py',
    '_base_/schedules/schedule_40k.py',
    '_base_/default_runtime.py',
    'mixins/classifier_coco_48_17.py',
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

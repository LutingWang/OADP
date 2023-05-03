trainer = dict(
    load_from='pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth',
    optimizer=dict(
        weight_decay=2.5e-5,
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
)

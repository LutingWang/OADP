model = dict(
    attn_weights_loss=dict(
        type='BCEWithLogitsLoss',
        weight=dict(
            type='WarmupScheduler',
            iter_=2000,
            value=2,
        ),
        gt_downsample='avg',
    ),
)

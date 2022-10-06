model = dict(
    attn_weights_loss=dict(
        type='BCEWithLogitsLoss',
        weight=dict(
            type='WarmupScheduler',
            iter_=1000,
        ),
        gt_downsample='avg',
    ),
)

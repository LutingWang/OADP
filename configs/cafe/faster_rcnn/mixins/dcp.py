model = dict(
    distiller=dict(
        losses=dict(
            loss_clip_patches=dict(
                type='MSELoss',
                norm=True,
                fields=['patches', 'clip_patches'],
                weight=dict(
                    type='WarmupScheduler',
                    value=100,
                    iter_=1000,
                ),
                reduction='mean',
            ),
        ),
    ),
)

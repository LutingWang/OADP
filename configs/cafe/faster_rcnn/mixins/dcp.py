model = dict(
    distiller=dict(
        adapts=dict(
            # patches=dict(
            #     type='Linear',
            #     in_features=512,
            #     out_features=512,
            #     fields=('patches',),
            # ),
        ),
        losses=dict(
            loss_clip_patches=dict(
                type='L1Loss',
                norm=True,
                fields=['patches', 'clip_patches'],
                weight=dict(
                    type='WarmupScheduler',
                    value=256,
                    iter_=200,
                ),
                # reduction='mean',
            ),
        ),
    ),
)

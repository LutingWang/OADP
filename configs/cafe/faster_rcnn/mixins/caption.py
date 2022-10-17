model = dict(
    distiller=dict(
        student_hooks=dict(
            feats=dict(
                type='StandardHook',
                path='_post_fpn',
            ),
        ),
        adapts=dict(
            adapted_feats=dict(
                type='Conv2d',
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                fields=('feats',),
                parallel=True,
            ),
        ),
        losses=dict(
            loss_clip_caption=dict(
                type='HierGKDLoss',
                fields=['adapted_feats', 'clip_captions'],
                weight=1,
            ),
        ),
    ),
)

model = dict(
    distiller=dict(
        student_hooks=dict(
            image=dict(
                type='StandardHook',
                path='._multilabel_classifier._linear',
            ),
        ),
        losses=dict(
            loss_clip_image=dict(
                type='MSELoss',
                norm=True,
                fields=['image', 'clip_image'],
                weight=dict(
                    type='WarmupScheduler',
                    value=2,
                    iter_=1000,
                ),
                reduction='sum',
            ),
        ),
    ),
)

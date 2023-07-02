model = dict(
    type='OADP',
    roi_head=dict(type='OADPRoIHead'),
    distiller=dict(
        student_hooks=dict(
            objects=dict(
                inputs=tuple(),
                action=dict(
                    type='StandardHook',
                    path='.roi_head._object_head.fc_cls._linear',
                ),
            ),
        ),
        adapts=dict(),
        losses=dict(
            loss_clip_objects=dict(
                inputs=('objects', 'clip_objects'),
                action=dict(
                    type='L1Loss',
                    weight=dict(type='WarmupScheduler', gain=256, end=200),
                ),
            ),
        ),
    ),
)

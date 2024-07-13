model = dict(
    global_head=dict(
        topk=20,
        classifier=dict(in_features=256),
        loss=dict(
            type='AsymmetricLoss',
            weight=dict(type='WarmupScheduler', gain=4, end=2000),
            gamma_neg=4,
            gamma_pos=0,
        ),
    ),
    distiller=dict(
        student_hook_pipeline=dict(
            global_=dict(
                type='SingleOperator',
                args=tuple(),
                atom=dict(
                    type=(
                        'TaskRegistry.KDRegistry.KDDistillerRegistry.'
                        'KDHookRegistry.Hook'
                    ),
                    path='._global_head._classifier._linear',
                ),
            ),
        ),
        loss_pipeline=dict(
            loss_clip_global=dict(
                type='SingleOperator',
                args=('global_', 'clip_global'),
                atom=dict(
                    type='ModelRegistry.LossRegistry.MSELoss',
                    weight=dict(type='WarmupScheduler', gain=0.5, end=200),
                    # FIXME: in todd, sum should multiply by number of GPUs
                    reduction='sum',
                ),
            ),
        ),
    ),
)

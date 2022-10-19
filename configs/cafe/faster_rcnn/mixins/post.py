model = dict(
    post_fpn=dict(
        refine_level=2,
        num_blocks=3,
        block=dict(
            channels=256,
            spatial_conv=dict(),
            task_attn=dict(),
            cross_attn=dict(
                num_heads=16,
                head_dims=32,
                avg_factor=3,
            ),
        ),
    ),
)

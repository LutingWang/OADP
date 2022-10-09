topK = 16
post_num_blocks = 3
model = dict(
    topK=topK,
    post_fpn=dict(
        refine_level=2,
        num_blocks=post_num_blocks,
        in_channels=256,
        block=dict(
            # glip_block=dict(
            #     num_heads=8,
            #     head_dims=32,
            #     avg_factor=post_num_blocks,
            # ),
            dyhead_block=dict(
                spatial_conv=dict(num_heads=topK),
                # scale_attn=dict(),
                # task_attn=dict(),
            ),
        ),
    ),
)

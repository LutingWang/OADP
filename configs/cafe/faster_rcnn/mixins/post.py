post_num_blocks = 3
model = dict(
    topK=20,
    hidden_dims=256,
    post_fpn=dict(
        refine_level=2,
        num_blocks=post_num_blocks,
        block=dict(
            # glip_block=dict(
            #     num_heads=8,
            #     head_dims=16,
            #     avg_factor=post_num_blocks,
            # ),
            dyhead_block=dict(
                spatial_conv=dict(),
                scale_attn=dict(),
                # task_attn=dict(),
            ),
        ),
    ),
)

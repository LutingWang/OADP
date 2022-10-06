model = dict(
    topK=20,
    hidden_dims=256,
    post_fpn=dict(
        refine_level=2,
        num_blocks=3,
        glip_block=dict(
            num_heads=8,
            head_dims=32,
        ),
    ),
)

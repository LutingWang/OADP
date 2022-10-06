model = dict(
    topK=20,
    hidden_dims=256,
    pre_fpn=dict(
        in_channels=[256, 512, 1024, 2048],
    ),
    neck=dict(
        in_channels=[256, 256, 256, 256],
    ),
)

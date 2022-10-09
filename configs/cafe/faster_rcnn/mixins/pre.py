model = dict(
    topK=16,
    pre_fpn=dict(
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
    ),
)

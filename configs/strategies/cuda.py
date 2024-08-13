trainer = dict(
    strategy=dict(type='CUDAStrategy'),
    dataloader=dict(sampler=dict(type='DistributedSampler', shuffle=True)),
)
validator = dict(
    strategy=dict(type='CUDAStrategy'),
    dataloader=dict(sampler=dict(type='DistributedSampler', shuffle=False)),
)

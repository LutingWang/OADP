_base_ = [
    'prompt.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            split=None,
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            split='ALL',
        ),
    ),
)

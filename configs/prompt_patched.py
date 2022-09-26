_base_ = [
    'prompt.py',
]

train = dict(
    dataloader=dict(
        dataset=dict(
            patched=True,
        ),
    ),
)
val = dict(
    dataloader=dict(
        dataset=dict(
            patched=True,
        ),
    ),
)

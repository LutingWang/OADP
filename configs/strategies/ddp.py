_base_ = [
    'cuda.py',
]

trainer = dict(strategy=dict(type='DDPStrategy'))
validator = dict(strategy=dict(type='DDPStrategy'))

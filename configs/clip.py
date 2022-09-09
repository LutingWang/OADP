data_root = 'data/coco/'
train = dict(
    batch_size=64,
    workers=8,
    dataset=dict(
        root=data_root + 'train2017',
        annFile=data_root + 'annotations/instances_train2017.json',
    ),
)
val = dict(
    batch_size=16,
    workers=2,
    dataset=dict(
        root=data_root + 'val2017',
        annFile=data_root + 'annotations/instances_val2017.json',
    ),
)

epoch = 40
lr = 2e-2
weight_decay = 1e-4

thr = 0.75
log_interval = 8

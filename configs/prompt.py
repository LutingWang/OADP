data_root = 'data/coco/'
patches_root = data_root + 'patches/'
train = dict(
    epoch=2,
    lr=5e-4,
    lr_scheduler=dict(
        milestones=[1],
    ),
    weight_decay=1e-4,

    batch_size=16,
    workers=2,
    dataset=dict(
        root=data_root + 'train2017',
        annFile=data_root + 'annotations/instances_train2017.json',
        patches_root=patches_root + 'train',
    ),
)
val = dict(
    batch_size=16,
    workers=2,
    dataset=dict(
        root=data_root + 'val2017',
        annFile=data_root + 'annotations/instances_val2017.json',
        patches_root=patches_root + 'val',
    ),
)

thr = 0.75
log_interval = 64

prompt = 'a photo of a'

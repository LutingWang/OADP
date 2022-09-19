data_root = 'data/coco/'
patches_root = data_root + 'patches/'
train = dict(
    batch_size=16,
    num_workers=4,
    patches_root=patches_root + 'train',
    dataset=dict(
        root=data_root + 'train2017',
        annFile=data_root + 'annotations/instances_train2017.json',
    ),
)
val = dict(
    batch_size=16,
    num_workers=4,
    patches_root=patches_root + 'val',
    dataset=dict(
        root=data_root + 'val2017',
        annFile=data_root + 'annotations/instances_val2017.json',
    ),
)

log_interval = 32

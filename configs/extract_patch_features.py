work_dir_root = 'work_dirs/patch_features_debug/'
data_root = 'data/coco/'
train = dict(
    batch_size=16,
    num_workers=4,
    work_dir=work_dir_root + 'train',
    dataset=dict(
        root=data_root + 'train2017',
        annFile=data_root + 'annotations/instances_train2017.json',
    ),
)
val = dict(
    batch_size=16,
    num_workers=4,
    work_dir=work_dir_root + 'val',
    dataset=dict(
        root=data_root + 'val2017',
        annFile=data_root + 'annotations/instances_val2017.json',
    ),
)

log_interval = 32

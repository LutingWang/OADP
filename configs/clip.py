image_size = 448

data_root = 'data/coco/'
train = dict(
    root=data_root + 'train2017',
    annFile=data_root + 'annotations/instances_train2017.json',
)
val = dict(
    root=data_root + 'val2017',
    annFile=data_root + 'annotations/instances_val2017.json',
)
batch_size = 64
workers = 8

epoch = 40
lr = 1e-2
weight_decay = 1e-4

thr = 0.75
log_interval = 8

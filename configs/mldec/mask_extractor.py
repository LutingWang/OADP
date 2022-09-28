data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=64,
        num_workers=4,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            pretrained='work_dirs/prompt/epoch_3_classes.pth',
            split='COCO_17',
        ),
    ),
)

log_interval = 64

model = dict(
    pretrained = 'pretrained/clip/ViT-B-32.pt',
)

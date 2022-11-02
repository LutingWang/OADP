data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=8,
        num_workers=0,
        dataset=dict(
            root=data_root+'val2017',
            ann_file=data_root+'annotations/instances_val2017.json',
            pretrained='data/epoch_3_classes.pth',
            split='COCO',
            mode = "longest_edge",
        ),
    ),
)

logger = dict(
    interval=64,
)
mini_batch_size = 512
model = dict(
    pretrained = 'ViT-B/32',
)

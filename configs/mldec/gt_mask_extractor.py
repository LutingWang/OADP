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
            mask_size = 7*2,
            expand_mode = "adaptive",
        ),
    ),
)

logger = dict(
    interval=64,
)
mini_batch_size = 512
model = dict(
    pretrained = 'ViT-B/32',
    patch_size=32,
    upsample=2,  # power of 2
)

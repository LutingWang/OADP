data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=64,
        num_workers=4,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            split='COCO_48',
            mask_size=7 * 2,
            expand_mode='adaptive',
        ),
    ),
)
train = dict(
    epoch=4,
    optimizer=dict(
        type='Adam',
        lr=5e-2,
        weight_decay=1e-3,
    ),
    dataloader=dict(
        batch_size=16,
        num_workers=4,
        dataset=dict(
            root=data_root + 'train2017',
            ann_file=data_root + 'annotations/instances_train2017.json',
            split='COCO_17',
            mask_size=7 * 2,  # upsample=2
            expand_mode='adaptive',
        ),
    ),
    loss=dict(
        type='CrossEntropyLoss',
        weight=8,
    ),
)

logger = dict(
    interval=4,
)
checkpoint = dict(
    load_=dict(
        model_config=dict(
            strict=False,
        ),
    ),
)

pretrained = 'pretrained/clip/ViT-B-32.pt'
model = dict(
    image_encoder=dict(
        patch_size=32,
        upsample=2,  # power of 2
    ),
    text_prompt=dict(
        prompt='a photo of a',
    ),
    text_encoder=dict(),
)

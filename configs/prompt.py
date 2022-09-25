data_root = 'data/coco/'
train = dict(
    epoch=4,
    optimizer=dict(
        type='Adam',
        lr=5e-2,
        weight_decay=1e-3,
    ),

    dataloader=dict(
        batch_size=1,
        workers=1,
        dataset=dict(
            root=data_root + 'train2017',
            ann_file=data_root + 'annotations/instances_train2017.json',
            split='COCO_48',
            # filter_empty=True,
        ),
    ),

    loss=dict(
        type='AsymmetricLoss',
        weight=640,
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    ),
)
val = dict(
    dataloader=dict(
        batch_size=1,
        workers=1,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            split='COCO_48',
            # filter_empty=True,
        ),
    ),
)

log_interval = 64

model = dict(
    text_prompt=dict(
        prompt='a photo of a',
    ),
    text_encoder=dict(),
    image_prompt=dict(
        length=8,
    ),
    image_encoder=dict(),
)

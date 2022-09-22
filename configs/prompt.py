data_root = 'data/coco/'
patches_root = data_root + 'patches/'
train = dict(
    epoch=4,
    optimizer=dict(
        type='Adam',
        params=[
            dict(params="(module.)?_(prompt|scaler|bias)", lr=5e-2),
            dict(params="(module.)?_image_refiner"),
        ],
        lr=5e-2,
        weight_decay=1e-4,
    ),
    lr_scheduler=dict(
        type='MultiStepLR',
        milestones=[3],
    ),

    batch_size=32,
    workers=4,
    dataset=dict(
        root=data_root + 'train2017',
        annFile=data_root + 'annotations/instances_train2017.json',
        patches_root=patches_root + 'train',
        split='COCO_48',
        filter_empty=True,
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
    batch_size=32,
    workers=4,
    dataset=dict(
        root=data_root + 'val2017',
        annFile=data_root + 'annotations/instances_val2017.json',
        patches_root=patches_root + 'val',
        split='COCO_17',
        filter_empty=True,
    ),
)

thr = 0.75
log_interval = 64

model = dict(
    prompt='a photo of a',
    image_refiner=dict(
        type='Base',
    ),
    # image_refiner=dict(
    #     type='TransformerEncoderLayer',
    #     num_channels=1024,
    #     num_heads=2,
    # ),
)

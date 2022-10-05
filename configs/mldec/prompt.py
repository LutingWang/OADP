data_root = 'data/coco/'
embeddings_root = data_root + 'embeddings/'
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
            embeddings_root=embeddings_root + 'train',
            mode='patch',
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
        batch_size=64,
        num_workers=4,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            embeddings_root=embeddings_root + 'val',
            mode='patch',
            split='COCO_17',
            # filter_empty=True,
        ),
    ),
)

logger = dict(
    interval=64,
)
checkpoint = dict(
    load_=dict(
        model_config=dict(
            strict=False,
        ),
    ),
)
model = dict(
    text_prompt=dict(
        prompt='a photo of a',
    ),
    text_encoder=dict(),
)
pretrained = 'pretrained/clip/ViT-B-32.pt'

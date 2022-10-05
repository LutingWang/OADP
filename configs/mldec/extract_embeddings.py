data_root = 'data/coco/'
embeddings_root = data_root + 'embeddings/'
train = dict(
    epoch=1,
    dataloader=dict(
        batch_size=1,
        num_workers=1,
        dataset=dict(
            root=data_root + 'train2017',
            ann_file=data_root + 'annotations/instances_train2017.json',
        ),
    )
)
val = dict(
    dataloader=dict(
        batch_size=1,
        num_workers=1,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
        ),
    )
)

logger = dict(
    interval=128,
)

model = dict(
    pretrained='pretrained/clip/ViT-B-32.pt',
)

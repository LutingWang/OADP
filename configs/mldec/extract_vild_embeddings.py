data_root = 'data/coco/'
embeddings_root = data_root + 'vild_embeddings/'
val = dict(
    dataloader=dict(
        batch_size=2,
        num_workers=2,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            proposal_file='work_dirs/proposal/val.pkl',
            # proposal_file='data/coco/proposals/rpn_r101_fpn_coco_val.pkl',
            embeddings_root=embeddings_root + 'val',
        ),
    ),
)
train = dict(
    epoch=1,
    dataloader=dict(
        batch_size=2,
        num_workers=2,
        dataset=dict(
            root=data_root + 'train2017',
            ann_file=data_root + 'annotations/instances_train2017.json',
            proposal_file='work_dirs/proposal/train.pkl',
            # proposal_file='data/coco/proposals/rpn_r101_fpn_coco_train.pkl',
            embeddings_root=embeddings_root + 'train',
        ),
    ),
)

logger = dict(
    interval=4,
)

model = dict(
    pretrained = 'pretrained/clip/ViT-B-32.pt',
)

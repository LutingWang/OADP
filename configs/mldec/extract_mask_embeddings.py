data_root = 'data/coco/'
embeddings_root = data_root + 'mask_embeddings/'
val = dict(
    dataloader=dict(
        batch_size=1,
        num_workers=1,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            # proposal_file='work_dirs/proposal/val.pkl',
            # proposal_sorted=True,
            # proposal_file='data/coco/proposals/rpn_r101_fpn_coco_val.pkl',
            # proposal_sorted=False,
            proposal_file=data_root + 'proposals/oln_r50_fpn_coco_val.pkl',
            proposal_sorted=True,
            mask_size=7 * 2,
            expand_mode='adaptive',
            embeddings_root=embeddings_root + 'val',
        ),
    ),
)
train = dict(
    epoch=1,
    dataloader=dict(
        batch_size=1,
        num_workers=1,
        dataset=dict(
            root=data_root + 'train2017',
            ann_file=data_root + 'annotations/instances_train2017.json',
            # proposal_file='work_dirs/proposal/train.pkl',
            # proposal_sorted=True,
            # proposal_file='data/coco/proposals/rpn_r101_fpn_coco_train.pkl',
            # proposal_sorted=False,
            proposal_file=data_root + 'proposals/oln_r50_fpn_coco_train.pkl',
            proposal_sorted=True,
            mask_size=7 * 2,  # upsample=2
            expand_mode='adaptive',
            embeddings_root=embeddings_root + 'train',
        ),
    ),
)

mini_batch_size = 512
logger = dict(
    interval=4,
)

model = dict(
    pretrained = 'pretrained/clip/ViT-B-32.pt',
    patch_size=32,
    upsample=2,  # power of 2
)

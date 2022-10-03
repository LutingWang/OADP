data_root = 'data/coco/'
embeddings_root = data_root + 'mask_embeddings/'
val = dict(
    dataloader=dict(
        batch_size=1,
        num_workers=0,
        dataset=dict(
            root=data_root + 'val2017',
            ann_file=data_root + 'annotations/instances_val2017.json',
            pretrained='data/epoch_3_classes.pth',
            split='COCO',
            proposal = '/mnt/data2/wlt/get_proposal/mmdetection/res_new.pkl',
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
        ),
    ),
)

log_interval = 64

model = dict(
    pretrained = 'ViT-B/32',
)

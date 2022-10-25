data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=8,
        num_workers=1,
        sample = True,
        dataset=dict(
            root=data_root+"train2017",
            ann_file=data_root+'annotations/instances_train2017.json',
            pretrained='data/coco/prompt/prompt1.pth',
            split='COCO_48_17',
            proposal = data_root + 'mask_embeddings/',
            top_KP = 100,
                
        ),
        ),
)
logger = dict(
    interval=64,
)
        
model = dict(
    dis = True,
    softmax_t = 1,
    # softmax_t = 1,
    topK_clip_scores = 1,
    nms_score_thres = 0.5,
    nms_iou_thres = 0.49705,
    bbox_objectness=dict(
        _name='mul',
        clip_score_ratio= 0.306,
        obj_score_ratio = 0.758,

    ),
)
json_name = "novel_pl_2.json"

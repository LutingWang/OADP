data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=8,
        num_workers=0,
        sample = True,
        dataset=dict(
            root=data_root+"train2017",
            ann_file=data_root+'annotations/instances_train2017.json',
            pretrained='data/coco/prompt/prompt2.pth',
            split='COCO_17',
            proposal = data_root + 'mask_embeddings/',
            top_KP = 99,
            lvis_ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
            lvis_split='LVIS'
                
        ),
        ),
)
logger = dict(
    interval=64,
)
        
model = dict(
    dis = True,
    softmax_t = 0.40192,
    # softmax_t = 1,
    topK_clip_scores = 1,
    nms_score_thres = 0.0065,
    nms_iou_thres = 0.6740,
    bbox_objectness=dict(
        _name='mul',
        clip_score_ratio= 0.71821,
        obj_score_ratio = 0.74209,

    ),
)
json_name = "lvis_pl.json"

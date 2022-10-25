data_root = 'data/coco/'
val = dict(
    dataloader=dict(
        batch_size=8,
        num_workers=0,
        sample = True,
        dataset=dict(
            root=data_root+"val2017",
            ann_file=data_root+'annotations/instances_val2017.json',
            pretrained='data/prompts/vild.pth',
            split='COCO_17',
            proposal = data_root + 'mask_embeddings/',
            top_KP = 100,
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
    softmax_t = 0.1,
    # softmax_t = 1,
    topK_clip_scores = 1,
    nms_score_thres = 0.0,
    nms_iou_thres = 0.5,
    bbox_objectness=dict(
        _name='mul',
        clip_score_ratio= 0.489887,
        obj_score_ratio = 0.79821,

    ),
)
json_name = "lvis_pl.json"
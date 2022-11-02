data_root = 'data/lvis_v1/'
val = dict(
    dataloader=dict(
        batch_size=8,
        num_workers=1,
        sample = True,
        dataset=dict(
            root=data_root+"val2017",
            ann_file=data_root+'annotations/lvis_v1_val.json',
            pretrained='data/prompts/vild.pth',
            split='LVIS',
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
    softmax_t = 0.01,
    # softmax_t = 1,
    topK_clip_scores = 1,
    nms_score_thres = 0.8,#0.5
    nms_iou_thres = 0.66,
    bbox_objectness=dict(
        _name='mul',
        clip_score_ratio= 0.488,
        obj_score_ratio = 0.457,

    ),
)
json_name = "lvis_novel_pl.json"

data_root = 'data/coco/'

dataloader = dict(
    batch_size=1,
    num_workers=1,
    sample=True,
    dataset=dict(
        root=data_root + "train2017/",
        ann_file=data_root + 'annotations/instances_train2017.48.json',
        proposal=data_root + 'oake/objects/train2017/',
        top_KP=99,
        classifier=dict(
            type="ViLDClassifier",
            pretrained='data/prompts/vild.pth',
        ),
    ),
)

logger = dict(interval=64)

generator = dict(
    softmax_t=0.40192,
    topK_clip_scores=1,
    nms_score_thres=0.007,
    nms_iou_thres=0.6740,
    clip_score_ratio=0.71821,
    obj_score_ratio=0.74209,
)

json_file = dict(type="coco", name="lvis_pl.json")

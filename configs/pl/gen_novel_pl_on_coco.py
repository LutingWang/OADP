data_root = 'data/coco/'

dataloader = dict(
    batch_size=8,
    num_workers=1,
    sample=True,
    dataset=dict(
        root=data_root + 'train2017/',
        ann_file=data_root + 'annotations/instances_train2017.48.json',
        proposal=data_root + 'oake/objects/train2017/',
        top_KP=100,
        classifier=dict(
            type="MLClassifier",
            pretrained='data/prompts/ml_coco.pth',
        ),
    ),
)

logger = dict(interval=64)

generator = dict(
    softmax_t=1,
    topK_clip_scores=1,
    nms_score_thres=0.5,
    nms_iou_thres=0.49705,
    clip_score_ratio=0.306,
    obj_score_ratio=0.758,
)

json_file = dict(type="lvis", name="novel_pl.json")

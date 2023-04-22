data_root = 'data/coco/'

dataloader = dict(
    batch_size=8,
    num_workers=1,
    sample=True,
    dataset=dict(
        type="lvis",
        root=data_root + "train2017",
        ann_file=data_root + 'annotations/instances_train2017.json',
        pretrained='data/prompts/vild.pth',
        proposal=data_root + 'oake/objects/',
        top_KP=99,
        lvis_ann_file='data/lvis_v1/annotations/lvis_v1_val.json',
        lvis_split='LVIS'
    ),
)

logger = dict(interval=64, )

model = dict(
    type="VLIDModel",
    dist=True,
    softmax_t=0.40192,
    topK_clip_scores=1,
    nms_score_thres=0.007,
    nms_iou_thres=0.6740,
    bbox_objectness=dict(
        _name='mul',
        clip_score_ratio=0.71821,
        obj_score_ratio=0.74209,
    ),
)

annotation_file = dict(type="lvis", name="lvis_pl.json")

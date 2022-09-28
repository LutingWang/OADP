_base_ = [
    'faster_rcnn_r50_fpn_1x_coco.py',
]

model = dict(
    type='Cafe',
    multilabel_classifier=dict(
        type='Classifier',
        pretrained='work_dirs/prompt/epoch_3_classes.pth',
        in_features=2048,
        out_features=80,
    ),
    multilabel_loss=dict(
        type='AsymmetricLoss',
        weight=dict(
            type='WarmupScheduler',
            value=16,
            iter_=1000,
        ),
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    ),
    topK=20,
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='work_dirs/prompt/epoch_3_classes.pth',
            ),
        ),
    ),
)

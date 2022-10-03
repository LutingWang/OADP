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
    hidden_dims=256,
    pre_fpn=dict(
        in_channels=[256, 512, 1024, 2048],
    ),
    neck=dict(
        in_channels=[256, 256, 256, 256],
    ),
    post_fpn=dict(
        refine_level=2,
        num_blocks=3,
        glip_block=dict(
            num_heads=8,
            head_dims=32,
        ),
    ),
    attn_weights_loss=dict(
        type='BCEWithLogitsLoss',
        weight=dict(
            type='WarmupScheduler',
            iter_=1000,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='work_dirs/prompt/epoch_3_classes.pth',
            ),
        ),
    ),
)

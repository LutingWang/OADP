model = dict(
    distiller=dict(
        student_hooks=dict(
            image=dict(
                type='StandardHook',
                path='._multilabel_classifier._linear',
            ),
        ),
        losses=dict(
            loss_clip_image=dict(
                type='MSELoss',
                norm=True,
                fields=['image', 'clip_image'],
                weight=dict(
                    type='WarmupScheduler',
                    value=0.5,
                    iter_=200,
                ),
                reduction='sum',
            ),
        ),
    ),
    multilabel_classifier=dict(
        type='Classifier',
        pretrained='data/coco/prompt/prompt1.pth',
        topK=20,
        in_features=256,
        out_features=80,
        split='COCO',
    ),
    multilabel_loss=dict(
        type='AsymmetricLoss',
        weight=dict(
            type='WarmupScheduler',
            value=8,
            iter_=1000,
        ),
        gamma_neg=4,
        gamma_pos=0,
        clip=0.05,
        disable_torch_grad_focal_loss=True,
    ),
)

model = dict(
    roi_head=dict(
        patch_head=dict(
            type='Shared2FCBBoxHead',
            with_reg=False,
            cls_predictor_cfg=dict(
                type='Classifier',
                pretrained='data/coco/prompt/prompt1.pth',
                split='COCO_48_17',
                num_base_classes=48,
            ),
            loss=dict(
                type='AsymmetricLoss',
                weight=dict(
                    type='WarmupScheduler',
                    value=32,
                    iter_=1000,
                ),
                gamma_neg=4,
                gamma_pos=0,
                clip=0.05,
                disable_torch_grad_focal_loss=True,
            ),
        ),
    ),
    distiller=dict(
        losses=dict(
            loss_clip_patches=dict(
                type='L1Loss',
                norm=True,
                fields=['patch_feats', 'clip_patch_feats'],
                weight=dict(
                    type='WarmupScheduler',
                    value=128,
                    iter_=200,
                ),
                # reduction='mean',
            ),

            loss_clip_patches_relation=dict(
                type='RKDLoss',
                fields=['patch_feats', 'clip_patch_feats'],
                weight=dict(
                    type='WarmupScheduler',
                    value=8,
                    iter_=200,
                ),
                # reduction='mean',
            ),

        ),
    ),
)

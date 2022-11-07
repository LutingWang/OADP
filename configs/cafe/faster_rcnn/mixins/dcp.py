model = dict(
    distiller=dict(
        adapts=dict(
            # patches=dict(
            #     type='Linear',
            #     in_features=512,
            #     out_features=512,
            #     fields=('patches',),
            # ),
        ),
        losses=dict(
            loss_clip_bboxes=dict(
                type='L1Loss',
                norm=True,
                fields=['bbox_feats', 'clip_bbox_feats'],
                weight=dict(
                    type='WarmupScheduler',
                    value=256,
                    iter_=200,
                ),
                # reduction='mean',
            ),

            loss_aux_clip_bboxes=dict(
                type='L1Loss',
                norm=True,
                fields=['aux_bbox_feats', 'aux_clip_bbox_feats'],
                weight=dict(
                    type='WarmupScheduler',
                    value=256,
                    iter_=200,
                ),
                # reduction='mean',
            ),

            # loss_clip_bboxes_relation=dict(
            #     type='RKDLoss',
            #     fields=['bbox_feats', 'clip_bbox_feats'],
            #     weight=dict(
            #         type='WarmupScheduler',
            #         value=8,
            #         iter_=200,
            #     ),
            #     # reduction='mean',
            # ),

        ),
    ),
)

_base_ = [
    'datasets/ov_lvis.py',
    'models/vild_ensemble_faster_rcnn_r50_fpn.py',
    'models/oadp.py',
    'models/block.py',
    'models/mask.py',
    'schedules/2x.py',
    'schedules/oadp.py',
    'runtime.py',
]

cls_predictor_cfg = dict(
    type='ViLDClassifier',
    prompts='data/prompts/detpro_lvis.pth',
    scaler=dict(
        train=0.01,
        val=0.007,
    ),
)
model = dict(
    # global_head=dict(
    #     classifier=dict(
    #         _delete_=True,
    #         type='ViLDClassifier',
    #         prompts='data/prompts/vild.pth',
    #         in_features=256,
    #         out_features=1203,
    #     ),
    # ),
    roi_head=dict(
        bbox_head=dict(cls_predictor_cfg=cls_predictor_cfg),
        object_head=dict(cls_predictor_cfg=cls_predictor_cfg),
        block_head=dict(cls_predictor_cfg=cls_predictor_cfg),
    ),
)

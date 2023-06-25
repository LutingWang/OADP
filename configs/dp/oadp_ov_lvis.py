_base_ = [
    'datasets/ov_lvis.py',
    # TODO: rename
    'models/oadp_faster_rcnn_r50_fpn_noblock.py',
    'models/mask.py',
    'schedules/2x.py',
    'schedules/oadp.py',
    'runtime.py',
]

model = dict(
    global_head=dict(
        classifier=dict(
            _delete_=True,
            type='ViLDClassifier',
            prompts='data/prompts/vild.pth',
            in_features=256,
            out_features=1203,
        ),
    ),
    roi_head=dict(
        type='ViLDEnsembleRoIHead',
        object_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
    ),
)

_base_ = [
    'datasets/ov_lvis.py',
    'models/oadp_faster_rcnn_r50_fpn_kd.py',
    'schedules/schedule_oadp_40k.py',
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
        object_head=dict(
            cls_predictor_cfg=dict(
                _delete_=True,
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                _delete_=True,
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
    ),
)

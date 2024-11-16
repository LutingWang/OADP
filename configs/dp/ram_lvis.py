_base_ = [
    'ov_lvis.py',
]

model = dict(
    roi_head=dict(
        type='RAMEnsembleOADPRoIHead',
        classifier_model=dict(
            type='RAMModel',
            ram_pred='data/lvis/annotations/ram_pred_results.pth'
        ),
    ),
)
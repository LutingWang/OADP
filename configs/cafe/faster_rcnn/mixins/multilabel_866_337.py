_base_ = [
    'multilabel.py',
]

model = dict(
    multilabel_classifier=dict(
        type='ViLDClassifier',
        pretrained='data/lvis_v1/prompt/detpro_lvis_1.pth',
        topK=50,
        out_features=1203,
        split='LVIS',
        num_base_classes=866,
    ),
)

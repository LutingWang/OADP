import copy

_base_ = "./_base_/grounding_dino_swin-t_pretrain_obj365.py"

text_detector = copy.deepcopy(_base_.model)
text_detector["test_cfg"] = dict(
    max_per_img=300,
    chunked_size=30,
)
image_detector = copy.deepcopy(_base_.model)
image_detector["type"] = "FSDetectorFeatureDistill"
image_detector["connector_cfg"] = dict(
    type="VisionProjector",
    visual_dim=3584,
    text_dim=768,
    hidden_dim=1024,
)
image_detector["test_cfg"] = dict(
    max_per_img=300,
    chunked_size=30,
)


model = dict(
    _delete_=True,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    type="EnsembledDetector",
    text_detector=text_detector,
    image_detector=image_detector,
)


lvis_label_map = "data/grounding_data/coco/annotations/lvis_v1_label_map.json"
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None, imdecode_backend="pillow"),
    dict(type="FixScaleResize", scale=(800, 1333), keep_ratio=True, backend="pillow"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="LoadEnsembleMMovodFeature",
        mmovod_pseudo_list="data/grounding_data/mmovod/pseudo_list.pth",
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "real_text",
            "pseudo_text",
            "custom_entities",
            "tokens_positive",
            "qwen_feature_list",
        ),
    ),
]

dataset_type = "LVISV1Dataset"
data_root = "data/grounding_data/coco/"

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file="annotations/lvis_v1_minival_inserted_image_name.json",
        data_prefix=dict(img=""),
        pipeline=test_pipeline,
        return_classes=True,
    )
)
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type="LVISFixedAPMetric",
    ann_file=data_root + "annotations/lvis_v1_minival_inserted_image_name.json",
)
test_evaluator = val_evaluator

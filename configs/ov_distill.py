import copy

_base_ = "./_base_/grounding_dino_swin-t_pretrain_obj365.py"

debug = False

model = dict(
    type="TextDetectorDistill",
    bbox_roi_extractor=dict(
        type="SingleRoIExtractor",
        roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    obj_loss_weight=0.025,
    block_loss_weight=0.1,
    global_loss_weight=0.025,
    test_cfg=dict(
        max_per_img=300,
        chunked_size=40,
    ),
    clip_hidden_dim=1024,
)

# dataset settings
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomChoiceResize",
        scales=[
            (480, 1333),
            (512, 1333),
            (544, 1333),
            (576, 1333),
            (608, 1333),
            (640, 1333),
            (672, 1333),
            (704, 1333),
            (736, 1333),
            (768, 1333),
            (800, 1333),
        ],
        keep_ratio=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type="RandomSamplingNegPos",
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256,
    ),
    dict(
        type="LoadEVAFeatures",
        dataset_name=None,
        data_dir="data/oadp",
        clip_version="eva-clip",
        longest_edge=False,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
            "flip",
            "flip_direction",
            "text",
            "custom_entities",
            "tokens_positive",
            "dataset_mode",
            "globals_features",
            "blocks_features",
            "objects_features",
        ),
    ),
]

# --------------------------- objects365 vg dataset---------------------------
objects365_pipeline = copy.deepcopy(train_pipeline)
objects365_pipeline[-2]["dataset_name"] = "o365"
objects365_dataset = dict(
    type="ODVGDataset",
    data_root="data/grounding_data/objects365/",
    ann_file='annotations/objects365_train.fs_od_exist.jsonl',
    label_map_file='annotations/o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=objects365_pipeline,
    return_classes=True,
    backend_args=None)

# --------------------------- flickr30k vg dataset---------------------------
flickr30k_pipeline = copy.deepcopy(train_pipeline)
flickr30k_pipeline[-2]["dataset_name"] = "flickr30k"
flickr30k_dataset = dict(
    type="ODVGDataset",
    data_root="data/grounding_data/flickr30k_entities/",
    ann_file="flickr_train_vg7.jsonl",
    label_map_file=None,
    data_prefix=dict(img="flickr30k_images/"),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=flickr30k_pipeline,
    return_classes=True,
    backend_args=None,
)

# --------------------------- gqa vg dataset---------------------------
debug_anns = "gqa_train_vg7.jsonl.subset"
gqa_pipeline = copy.deepcopy(train_pipeline)
gqa_pipeline[-2]["dataset_name"] = "gqa"
gqa_dataset = dict(
    type="ODVGDataset",
    data_root="data/grounding_data/gqa/",
    ann_file="gqa_train_vg7.jsonl" if not debug else debug_anns,
    label_map_file=None,
    data_prefix=dict(img="images/"),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=gqa_pipeline,
    return_classes=True,
    backend_args=None,
)

# --------------------------- llava vg dataset---------------------------
llava_pipeline = copy.deepcopy(train_pipeline)
llava_pipeline[-2]["dataset_name"] = "llava-cap"
caption_dataset = dict(
    type="ODVGDataset",
    data_root="data/grounding_data/llava_cap/",
    ann_file="LLaVA-ReCap-558K_tag_box_vg7.jsonl",
    label_map_file=None,
    data_prefix=dict(img="images/"),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=llava_pipeline,
    return_classes=True,
    backend_args=None,
)

# --------------------------- v3det vg dataset---------------------------
v3det_pipeline = copy.deepcopy(train_pipeline)
v3det_pipeline[-2]["dataset_name"] = "v3det"
v3det_dataset = dict(
    type="ODVGDataset",
    data_root="data/grounding_data/v3det/",
    ann_file="annotations/v3det_2023_v1_train_vg7.jsonl",
    label_map_file=None,
    data_prefix=dict(img=""),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=v3det_pipeline,
    return_classes=True,
    backend_args=None,
)

# --------------------------- v3det od dataset---------------------------
v3det_od_pipeline = copy.deepcopy(train_pipeline)
v3det_od_pipeline[-2]["dataset_name"] = "v3det"
v3det_od_pipeline[-3]["label_map_file"] = "data/grounding_data/v3det/annotations/v3det_2023_v1_label_map.json"
v3det_od_dataset = dict(
    type='ODVGDataset',
    data_root='data/grounding_data/v3det/',
    ann_file='annotations/v3det_2023_v1_train_od.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,
    pipeline=v3det_od_pipeline,
    return_classes=True,
    backend_args=None)


debug_datasets = [objects365_dataset]
train_datasets = [gqa_dataset, flickr30k_dataset, v3det_dataset, objects365_dataset]
train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="ConcatDataset", datasets=train_datasets if not debug else debug_datasets
    ),
)

dataset_type = "LVISV1Dataset"
data_root = "data/grounding_data/coco/"

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file="annotations/lvis_v1_minival_inserted_image_name.json",
        data_prefix=dict(img=""),
        pipeline=_base_.test_pipeline,
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

optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
            "language_model": dict(lr_mult=0.1),
            # 'lmm': dict(lr_mult=0.1),
        }
    ),
)


max_iter = 150000
train_cfg = dict(
    _delete_=True, type="IterBasedTrainLoop", max_iters=max_iter, val_interval=150000
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[120000, 140000],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=10000, max_keep_ckpts=30),
    visualization=dict(type="GroundingVisualizationHook"),
    logger=dict(type="LoggerHook", interval=100),
)
log_processor = dict(by_epoch=False)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=True)

# default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))

env_cfg = dict(
    dist_cfg=dict(backend="nccl", timeout=36000),  # 36000s = 10h
)

load_from = "data/huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth"

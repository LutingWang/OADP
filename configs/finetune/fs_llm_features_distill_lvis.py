import copy
_base_ = '../_base_/grounding_dino_swin-t_pretrain_obj365.py'

data_root = 'data/grounding_data/coco/'

model = dict(
    type='FSDetectorFeatureDistill',
    connector_cfg=dict(
        type='VisionProjector',
        visual_dim=3584,
        text_dim=768,
        hidden_dim=1024
    ),
    test_cfg=dict(
        max_per_img=300,
        chunked_size=30,
    )
)

# =============================== train dataset ===============================

qwen_feature_map = 'data/grounding_data/mmovod/pseudo_list.pth'
label_map = 'data/grounding_data/mmovod/pseudo_dict.json'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPosPseudoLabel',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        label_map_file=label_map,
        max_tokens=256
    ),
    dict(type='LoadMMovodFinetuneFeature', 
        mmovod_pseudo_list=qwen_feature_map,
        label_map=label_map),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode', 
                   'qwen_feature_list', 'text_list'))
] 

lvis_dataset=dict(
    type='ClassBalancedDataset',
    oversample_thr=1e-3,
    dataset=dict(
        type='ODVGDataset',
        data_root=data_root,
        need_text=False,
        label_map_file='annotations/lvis_v1_label_map_norare.json',
        ann_file='annotations/lvis_v1_train_od_norare.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=lvis_dataset)

# =============================== test dataset ===============================
test_pipeline = [
    dict(
        type='LoadImageFromFile', backend_args=None,
        imdecode_backend='pillow'),
    dict(
        type='FixScaleResize',
        scale=(800, 1333),
        keep_ratio=True,
        backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadMMovodFeature', mmovod_pseudo_list='data/grounding_data/mmovod/pseudo_list.pth'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive', 'qwen_feature_list'))
]

dataset_type = 'LVISV1Dataset'
data_root = 'data/grounding_data/coco/'

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        type=dataset_type,
        ann_file='annotations/lvis_v1_val.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline, 
        return_classes=True))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_val.json')
test_evaluator = val_evaluator


# =============================== runtime ===============================
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
            # 'lmm': dict(lr_mult=0.1),
        }))

max_iter = 26000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=max_iter)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[60000, 80000],
        gamma=0.1)
]

default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000),
    # visualization=dict(type='GroundingVisualizationHook', draw=True, test_out_dir='images'),
    logger=dict(type='LoggerHook', interval=100))
log_processor = dict(by_epoch=False)

auto_scale_lr = dict(base_batch_size=64, enable=True)


load_from = 'work_dirs/fs_llm_features_distill_0.00_0.8_0.0/iter_16000.pth'
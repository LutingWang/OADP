import copy
_base_ = './_base_/grounding_dino_swin-t_pretrain_obj365.py'

debug = False

model = dict(
    type='FSDetectorFeatureDistill',
    connector_cfg=dict(
        type='VisionProjector',
        visual_dim=3584,
        text_dim=768,
        hidden_dim=1024
    ),
    w_distill=0.05,
    w_global=0.8,
    w_structure=0.8,
    test_cfg=dict(
        max_per_img=300,
        chunked_size=30,
    )
)

# =============================== train dataset ===============================

qwen_feature_map = 'data/grounding_data/qwen/annotations/name2pth.json'
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
        label_map_file=None,
        max_tokens=256),
    dict(type='LoadQwenFeature', 
        qwen_feature_map=qwen_feature_map,
        label_map=None,
        use_gt_name=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode', 
                   'qwen_feature_list', 'text_list'))
]


v3det_pseudo_label_map = 'data/grounding_data/v3det/annotations/pseudo_label_map.json'
v3det_label_map = 'data/grounding_data/v3det/annotations/v3det_2023_v1_label_map_refine.json'
debug_v3det_ann = 'annotations/v3det_2023_v1_train.fs_od.subset.json'
v3det_train_pipeline = copy.deepcopy(train_pipeline)
v3det_train_pipeline[-3]['label_map_file'] = v3det_pseudo_label_map
v3det_train_pipeline[-2]['label_map'] = v3det_label_map
v3det_dataset = dict(
    type='ODVGDataset',
    data_root='data/grounding_data/v3det/',
    ann_file='annotations/v3det_2023_v1_train.fs_od.json' if not debug else debug_v3det_ann,
    label_map_file='annotations/pseudo_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=v3det_train_pipeline,
    return_classes=True,
    backend_args=None)

objects365_pseudo_label_map = 'data/grounding_data/objects365/annotations/pseudo_label_map.json'
objects365_label_map = 'data/grounding_data/objects365/annotations/objects365_label_map_refine.json'
objects365_train_pipeline = copy.deepcopy(train_pipeline)
objects365_train_pipeline[-3]['label_map_file'] = objects365_pseudo_label_map
objects365_train_pipeline[-2]['label_map'] = objects365_label_map
objects365_dataset = dict(
    type='ODVGDataset',
    data_root='data/grounding_data/objects365/',
    ann_file='annotations/objects365_train.fs_od.json',
    label_map_file='annotations/pseudo_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=objects365_train_pipeline,
    return_classes=True,
    backend_args=None)


if debug:
    dataset = dict(type='ConcatDataset', datasets=[v3det_dataset])
else:
    dataset = dict(type='ConcatDataset', datasets=[v3det_dataset, objects365_dataset])

train_dataloader = dict(
    _delete_=True,
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dataset)

# =============================== test dataset ===============================
lvis_label_map = 'data/grounding_data/coco/annotations/lvis_v1_label_map.json'
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
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img=''),
        pipeline=test_pipeline, 
        return_classes=True))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator

if debug:
    test_dataloader = val_dataloader = None
    test_evaluator = val_evaluator = None
    test_cfg = val_cfg = None

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

max_iter = 16000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=max_iter//4)

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


load_from = 'data/huggingface/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth'
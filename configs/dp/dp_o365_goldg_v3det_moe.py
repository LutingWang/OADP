_base_ = '../gd_pretrain/grounding_dino_swin-t_pretrain_obj365.py'
server_root = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/'
load_from = server_root + 'ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_moe.pth'

model = dict(
    type='DPGroundingDino',
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    moe_cfg=dict(
        expert_num=8,
        inputs_dim=256,
        topk=2,
    )
)


v3det_post_transform = [
    dict(
        type='LoadFeature', 
        pth_dir=server_root+'oake/v3det/', 
        data_root=server_root+'data/V3Det/'
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'blocks_features', 'globals_features', 'objects_features'))
]

o365v1_post_transform = [
    dict(
        type='LoadFeature', 
        pth_dir=server_root+'oake/objects365v1/', 
        data_root=server_root+'data/objects365v1/train/'
    ),
]

flickr30k_post_transform = [
    dict(
        type='LoadFeature', 
        pth_dir=server_root+'oake/flickr/', 
        data_root=server_root+'data/flickr30k_entities/flickr30k_images/'
    ),
]

gqa_post_transform = [
    dict(
        type='LoadFeature', 
        pth_dir=server_root+'oake/gqa/', 
        data_root=server_root+'data/gqa/images/'
    ),
]

o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/objects365v1/',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline[:-1] + o365v1_post_transform + [v3det_post_transform[-1]],
    return_classes=True,
    backend_args=None,
)

flickr30k_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/flickr30k_entities/',
    ann_file='final_flickr_separateGT_train_vg.json',
    label_map_file=None,
    data_prefix=dict(img='flickr30k_images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline[:-1] + flickr30k_post_transform + [v3det_post_transform[-1]],
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline[:-1] + gqa_post_transform + [v3det_post_transform[-1]],
    return_classes=True,
    backend_args=None)

v3d_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file=server_root+'data/V3Det/annotations/v3det_2023_v1_label_map.json',
        max_tokens=256),
]
v3det_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/V3Det/',
    ann_file='annotations/v3det_2023_v1_train_od.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,  # change this
    pipeline=v3d_train_pipeline + v3det_post_transform,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    dataset=dict(datasets=[
        o365v1_od_dataset, flickr30k_dataset, gqa_dataset, v3det_dataset
    ]))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004 * 0.1 * 0.1,
                   weight_decay=0.0001),  # bs=16 0.0001
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0.1),
        }))

# learning policy
iter_per_epoch = 12196
max_iter = 2 * iter_per_epoch

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=1000),
]

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=iter_per_epoch)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)

default_hooks = dict(visualization=dict(type='GroundingVisualizationHook'))
custom_hooks = [dict(type='CheckpointHook', by_epoch=False, interval=iter_per_epoch)]
find_unused_parameters = True
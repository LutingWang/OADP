_base_ = './grounding_dino_swin-t_pretrain_obj365.py'
server_root = '/data/yhq/OADP/'
lang_model_name = 'pretrained/google-bert/bert-base-uncased'
pretrained = server_root + 'pretrained/swin_tiny_patch4_window7_224.pth'
load_from = server_root + 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_moe.pth'

model = dict(
    type='DPGroundingDino',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
    ),
    language_model = dict(
        name=lang_model_name,
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32],
    ),
    moe_cfg=dict(
        expert_num=4,
        inputs_dim=256,
        topk=2,
    )
)

o365v1_od_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/objects365v1/',
    ann_file='objects365_train_od.json',
    label_map_file='o365v1_label_map.json',
    data_prefix=dict(img='train/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
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
    pipeline=_base_.train_pipeline,
    return_classes=True,
    backend_args=None)

gqa_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/gqa/',
    ann_file='final_mixed_train_no_coco_vg.json',
    label_map_file=None,
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=_base_.train_pipeline,
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
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        # change this
        label_map_file=server_root+'data/V3Det/annotations/v3det_2023_v1_label_map.json',
        max_tokens=256),
    dict(type='LoadFeature', pth_dir=server_root+'work_dirs/oake/v3det/', data_root=server_root+'data/V3Det/'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'blocks_features', 'globals_features', 'objects_features'))
]
v3det_dataset = dict(
    type='ODVGDataset',
    data_root=server_root+'data/V3Det/',
    ann_file='annotations/v3det_2023_v1_train_od_test.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,  # change this
    pipeline=v3d_train_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    dataset=dict(datasets=[
        # o365v1_od_dataset, flickr30k_dataset, gqa_dataset, v3det_dataset
        v3det_dataset
    ]))
test_dataloader = val_dataloader = None
test_cfg = val_cfg = None
val_evaluator = test_evaluator = None
custom_hooks = [dict(type='CheckpointHook', by_epoch=False, interval=1)]
find_unused_parameters = True
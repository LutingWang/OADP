_base_ = './grounding_dino_swin-t_pretrain_obj365.py'
# load_from = _base_.server_root + 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'

model = dict(
    type='FsGroundingDINO',
    fs_model_cfg=dict(
        type='FewShotModel',
        language_model_cfg=dict(
            type='BertModel',
            name=_base_.lang_model_name,
            max_tokens=256,
            pad_to_max=False,
            use_sub_sentence_represent=True,
            special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
            add_pooling_layer=False,
        ),
        clip_model_path='pretrained/clip/ViT-B-32.pt',
        vision_agg_cfg=dict(
            type='VisualAggregatorT',
            max_shots=10,
            d_model=512,
            layers=12,
            heads=8,
        ),
        loss_type='contrastive',
        projector_type='mlp2x_gelu'
    )
)

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
        type='SampleRefImages', 
        max_sample_num=40,
        min_imgs=5,
        max_imgs=5,
        label_map_path=_base_.server_root+'data/V3Det/annotations/v3det_imagenet_label_map.json',
        samples_data_root=_base_.server_root+'data/imagenet21k/images',
        samples_label_map=_base_.server_root+'data/imagenet21k/annotations/imagenet21k_label2images.json'
    ),
    dict(type='ClipTransform', in_key='ref_images', out_key='ref_images'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'ref_images', 'n_images', 'n_samples'),)
]

v3det_dataset = dict(
    type='FsODVGDataset',
    data_root=_base_.server_root+'data/V3Det/',
    ann_file='annotations/v3det_2023_v1_train_od_test.json',
    label_map_file='annotations/v3det_2023_v1_label_map.json',
    imagenet_label_map='annotations/v3det_imagenet_label_map.json',
    data_prefix=dict(img=''),
    filter_cfg=dict(filter_empty_gt=False),
    need_text=False,  # change this
    pipeline=v3d_train_pipeline,
    return_classes=True,
    backend_args=None)

train_dataloader = dict(
    batch_size=2,
    dataset=dict(datasets=[
        # o365v1_od_dataset, flickr30k_dataset, gqa_dataset, v3det_dataset
        v3det_dataset
    ]))


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
    dict(
        type='SampleRefImages',
        min_imgs=5,
        max_imgs=5,
        label_map_path=_base_.server_root+'data/lvis/annotations/mmovod_label_map.json',
        samples_data_root=_base_.server_root+'data/mmovod-samples/images',
        samples_label_map=_base_.server_root+'data/mmovod-samples/annotations.json'
    ),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]


custom_hooks = [dict(type='CheckpointHook', by_epoch=False, interval=1)]
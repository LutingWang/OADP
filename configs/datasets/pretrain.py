root = '/mnt/data1/wlt/workspace/OADP/'
split = '_mini'
# split = ''
train_batch_size_per_gpu = 2
test_batch_size_per_gpu = 1


affine_scale = 0.9
img_scale = (1333, 800)
num_classes = 1203
num_training_classes = 80

pre_transform = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

last_transform = [
    # Delete gt_masks to avoid more computation
    dict(type='mmyolo.RemoveDataElement', keys=['gt_masks']),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='mmyolo.YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
]

text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]

train_pipeline_stage1 = [
    *pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=img_scale,
         pad_val=114.0,
         pre_transform=pre_transform),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform,
    *text_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='mmyolo.YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=100,
        border_val=(114, 114, 114)), 
    *last_transform,
    *text_transform
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='mmyolo.YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='mmyolo.LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114),
    ),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]

obj365v1_train = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V2Dataset',
        data_root=f'{root}data/objects365v2/',
        ann_file=f'annotations/zhiyuan_objv2_train{split}.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=f'{root}data/texts/obj365v2_class_texts.json',
    pipeline=train_pipeline_stage1)

mixgrounding_train = dict(type='YOLOv5MixedGroundingDataset',
                        data_root=f'{root}data/mixed_grounding/',
                        ann_file=f'annotations/final_mixed_train_no_coco{split}.json',
                        data_prefix=dict(img='images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline_stage1)

flickr_train = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root=f'{root}data/flickr/',
    ann_file=f'annotations/final_flickr_separateGT_train{split}.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline_stage1)

train_dataset = dict(
    type='ConcatDataset',
    datasets=[
        obj365v1_train, 
        # mixgrounding_train, 
        # flickr_train
    ],
    ignore_keys=['classes', 'palette']
)


val_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5LVISV1Dataset',
        data_root=f'{root}data/lvis/',
        test_mode=True,
        ann_file=f'annotations/lvis_v1_minival_inserted_image_name{split}.json',
        data_prefix=dict(img=''),
        batch_shapes_cfg=None
    ),
    class_text_path=f'{root}data/texts/lvis_v1_class_texts.json',
    pipeline=val_pipeline
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=train_dataset
)

val_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=val_dataset   
)

val_evaluator = dict(
    type='mmdet.LVISMetric',
    ann_file=f'{root}data/lvis/annotations/lvis_v1_minival_inserted_image_name{split}.json',
    metric='bbox'
)
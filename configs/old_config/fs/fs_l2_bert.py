_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
server_root = ''
lang_model_name = "pretrained/google-bert/bert-base-uncased"
load_from = 'pretrained/oadp/fs/language_model.pth'

train_pipeline = [
    dict(type='RandomLoadFromFile', min_imgs=2, max_imgs=10),
    dict(type='ClipTransform', n_px=224),
    dict(type='CleanText'),
    dict(type='PackFsData'),
]

imagenet21k = dict(
    type='ImageNet21KDataset',
    data_root=server_root+'data/imagenet21k/',
    ann_file='annotations/imagenet21k_labels_test.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='fs_collect_fn'),
    dataset=imagenet21k,
)


model = dict(
    type='FewShotModelBert',
    language_model_cfg=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?', '[image]'],
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
    loss_type='l2',
    projector_type='mlp2x_gelu'
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=32)
val_cfg = test_cfg = None

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=1e-4, weight_decay=0.02))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=12),
    dict(
        type='MultiStepLR',
        begin=0,
        end=32,
        by_epoch=True,
        milestones=[20],
        gamma=0.1)
]

visualizer = dict(
    type="Visualizer",
    vis_backends=[
        dict(type="TensorboardVisBackend"),
        # dict(type='WandbVisBackend',
        #  init_kwargs=dict(
        #     project='OADP',
        #     name='fs-t-contrastive',
        #  ))
    ],
)
# https://docs.wandb.ai/ref/python/init/

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, interval=1, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=10, type='LoggerHook'),
)
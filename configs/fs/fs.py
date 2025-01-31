_base_ = ['../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py']
server_root = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/'
lang_model_name = "pretrained/google-bert/bert-base-uncased"

train_pipeline = [
    dict(type='RandomLoadFromFile', min_imgs=2, max_imgs=10),
    dict(type='ClipTransform', n_px=224),
    dict(type='CleanText'),
    dict(type='PackFsData'),
]

imagenet21k = dict(
    type='ImageNet21KDataset',
    data_root=server_root+'data/imagenet21k/',
    ann_file='annotations/imagenet21k_labels.json',
    data_prefix=dict(img='images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline
)

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='fs_collect_fn'),
    dataset=imagenet21k,
)


model = dict(
    type='FewShotModel',
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        max_tokens=256,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False,
    ),
    clip_model_path='pretrained/clip/ViT-B-32.pt',
)

val_cfg = test_cfg = None
_base_ = '../_base_/grounding_dino_swin-t_pretrain_obj365.py'

model = dict(test_cfg=dict(
    max_per_img=300,
    chunked_size=30,
))

dataset_type = 'LVISV1Dataset'
data_root = 'data/grounding_data/coco/'

fs_pipeline = _base_.test_pipeline[:-1] + [
    dict(type='SubLabel', label_map='data/grounding_data/coco/annotations/Intern1b_mapping.json')
] + _base_.test_pipeline[-1:]

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=fs_pipeline,
        type=dataset_type,
        ann_file='annotations/lvis_v1_minival_inserted_image_name.json',
        data_prefix=dict(img='')))
test_dataloader = val_dataloader

# numpy < 1.24.0
val_evaluator = dict(
    _delete_=True,
    type='LVISFixedAPMetric',
    ann_file=data_root +
    'annotations/lvis_v1_minival_inserted_image_name.json')
test_evaluator = val_evaluator

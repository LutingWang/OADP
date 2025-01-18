_base_ = '../gd_pretrain/grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py'
server_root = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/'
# load_from = server_root + 'ckpt/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0004,
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
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[iter_per_epoch, 2 * iter_per_epoch],
        gamma=0.1
    )
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
pipenv run bash tools/torchrun.sh -m oadp.dp.train \
    oadp_fs_lvis_ensemble \
    configs/dp/oadp_ov_lvis.py \
    --load-model-from pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth

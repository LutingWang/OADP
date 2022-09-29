import argparse
import pathlib
import sys

import einops
import todd
import torch
import torch.distributed
from mmdet.models import build_detector
from mmdet.datasets import CocoDataset, build_dataset
import sklearn.metrics
from tqdm import trange

sys.path.insert(0, '')
from cafe import Cafe, one_hot
from mldec import odps_init, debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--load', required=True)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    debug.init()
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    dataset_config = config.data.val.copy()
    dataset_config.update(pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type='Pad', size_divisor=32),
        dict(type='ImageToTensor', keys=['img']),
        dict(type='ToTensor', keys=['gt_labels']),
        dict(type='Collect', keys=['img', 'gt_labels'], meta_keys=tuple()),
    ])
    dataset = build_dataset(dataset_config)

    ckpt = torch.load(args.load, 'cpu')
    model: Cafe = build_detector(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'),
    )
    model.eval()
    model.requires_grad_(False)
    model.load_state_dict(ckpt['state_dict'])
    if not debug.CPU:
        model = model.cuda()

    results = []
    for i in trange(len(dataset)):
        sample = dataset[i]
        img = sample['img']
        if not debug.CPU:
            img = img.cuda()
        x = img.unsqueeze(0)
        feats = model.backbone(x)
        multilabel_logits = model.multilabel_classify(feats)
        results.append((multilabel_logits, sample['gt_labels']))

    multilabel_logits_, gt_labels = zip(*results)
    multilabel_logits = torch.cat(multilabel_logits_)
    img_labels = one_hot(gt_labels, model._multilabel_classifier.num_classes)
    mAP = sklearn.metrics.average_precision_score(img_labels.cpu().numpy(), multilabel_logits.cpu().numpy(), average=None)
    topK_recall = model.topK(multilabel_logits, img_labels, average=None)
    for class_, mAP_, topK_recall_ in zip(CocoDataset.CLASSES, mAP, topK_recall):
        print(f"{class_} mAP={mAP_} recall={topK_recall_}")
    print(f"mAP {mAP.mean()}, topK_recall {topK_recall.mean()}")

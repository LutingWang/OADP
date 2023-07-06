# type: ignore
import argparse
import os
import pathlib
from typing import Any

import numpy as np
import todd
import torch
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

from oadp.base import Categories, Globals, coco, lvis  # noqa: F401
from oadp.prompts.vild import gen_prompts

from .utils import plot_single_img


def set_custom_category(
    work_dir: str, config: todd.Config, categories: Categories
):
    # set categories
    Globals.training = False
    Globals.categories = categories

    # generate prompt
    prompt_path = os.path.join(work_dir, 'new_prompt.pth')
    gen_prompts(categories, prompt_path)

    # override config
    config.model.global_head.classifier.type = 'ViLDClassifier'
    config.model.roi_head.bbox_head.cls_predictor_cfg.type = 'ViLDClassifier'
    config.model.roi_head.object_head.cls_predictor_cfg.type = 'ViLDClassifier'
    config.model.roi_head.block_head.cls_predictor_cfg.type = 'ViLDClassifier'

    config.model.global_head.classifier.prompts = prompt_path
    config.model.roi_head.bbox_head.cls_predictor_cfg.prompts = prompt_path
    config.model.roi_head.object_head.cls_predictor_cfg.prompts = prompt_path
    config.model.roi_head.block_head.cls_predictor_cfg.prompts = prompt_path
    config.model.global_head.classifier.out_features = categories.num_all


def single_gpu_infer(model: Any, imgs: list[str] | str,
                     config: todd.Config) -> list[Any]:
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    validator: todd.Config = config.validator.dataloader
    validator.dataset.pipeline = replace_ImageToTensor(
        validator.dataset.pipeline
    )
    test_pipeline = Compose(validator.dataset.pipeline)

    data_list = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            data = dict(img=img)
        else:
            data = dict(img_info=dict(filename=img), img_prefix=None)
        data = test_pipeline(data)
        data_list.append(data)

    data = collate(data_list, samples_per_gpu=len(imgs))
    data['img_metas'] = [metas.data[0] for metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    if not is_batch:
        return results[0]
    else:
        return results


def build_model(config: todd.Config, checkpoint: str) -> torch.nn.Module:
    config.pop('train_cfg', None)
    config.pop('pretrained', None)
    backbone: todd.Config = config.backbone
    backbone.pop('init_cfg', None)
    model = build_detector(config)
    load_checkpoint(model, checkpoint, map_location='cpu')
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=pathlib.Path)
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image')

    parser.add_argument('--override', action=todd.DictAction)
    parser.add_argument('--category', help='new category')
    args = parser.parse_args()
    return args


def main():
    # parse args
    args = parse_args()
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)

    # make work dir
    name: pathlib.Path = args.name
    work_dir = 'work_dirs' / name
    work_dir.mkdir(parents=True, exist_ok=True)

    # set GPU settings
    if todd.Store.CUDA:
        torch.distributed.init_process_group('nccl')
        torch.cuda.set_device(todd.get_local_rank())

    # set category
    if args.category is not None:
        new_categories = Categories(
            bases=(), novels=set(args.category.split(','))
        )
        set_custom_category(work_dir, config, new_categories)
    else:
        Globals.categories = eval(config.categories)

    # build model
    model = build_model(config.model, args.checkpoint)

    # inference
    results = single_gpu_infer(model, args.img, config)

    # plot
    plot_single_img(
        args.img, results, 0.3, str(work_dir / 'result.png'),
        Globals.categories.all_
    )


if __name__ == "__main__":
    main()

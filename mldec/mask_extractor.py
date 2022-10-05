import argparse
from collections import namedtuple
import pathlib
from typing import Any, Dict, Iterator, List, Optional, Tuple
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed
import torch.nn as nn
import PIL.Image

import clip
import clip.model

import todd
import torchvision

from . import datasets

from .debug import debug
from .utils import all_gather, odps_init
from .todd import BaseRunner

Batch = namedtuple(
    'Batch',
    ['images', 'masks', 'bboxes', 'bbox_labels', 'class_embeddings', 'scaler', 'bias'],
)


def prep(
    image: PIL.Image.Image,
    bboxes: torch.Tensor,
    bbox_labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        image: original image
        bboxes: n x 4
        bbox_labels: n

    Returns:
        images: n' x 3 x 224 x 224, cropped bboxes
        masks: n' x 197, processed transformer masks
        bboxes: n' x 4, filtered bboxes
        bbox_labels: n', filtered labels
    """
    raise NotImplementedError


class CocoClassification(torchvision.datasets.CocoDetection):
    _classnames: List[str]
    _cat2label: Dict[int, int]

    def __init__(
        self,
        root: str,
        ann_file: str,
        pretrained: str,
        split: Optional[str] = None,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )

        if split is None:
            self._classnames = [cat['name'] for cat in self.coco.cats.values()]
            self._cat2label = {cat: i for i, cat in enumerate(self.coco.cats)}
        else:
            classnames = getattr(datasets, split)
            self._classnames = []
            self._cat2label = dict()
            for cat in self.coco.cats.values():
                if cat['name'] in classnames:
                    self._classnames.append(cat['name'])
                    self._cat2label[cat['id']] = len(self._cat2label)

        ckpt = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = ckpt['embeddings']
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in self._classnames]
        self._class_embeddings = embeddings[inds]
        self._scaler = ckpt['scaler'].item()
        self._bias = ckpt['bias'].item()

    @property
    def classnames(self) -> Tuple[str]:
        return tuple(self._classnames)

    @property
    def num_classes(self) -> int:
        return len(self._classnames)

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [anno for anno in target if anno['category_id'] in self._cat2label]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image, target = super().__getitem__(index)
        bboxes = torch.tensor([anno['bbox'] for anno in target])
        bbox_labels = torch.tensor([self._cat2label[anno['category_id']] for anno in target])
        return prep(image, bboxes, bbox_labels)

    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
        images, masks, bboxes, bbox_labels = map(torch.cat, zip(*batch))
        return Batch(images, masks, bboxes, bbox_labels, self._class_embeddings, self._scaler, self._bias)


class Model(todd.base.Module):

    def __init__(
        self,
        *args,
        config: todd.base.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        clip_model, _ = clip.load(config.pretrained, 'cpu')
        clip_model.requires_grad_(False)
        raise NotImplementedError

    def forward(self, batch: Batch) -> torch.Tensor:
        raise NotImplementedError


class Runner(BaseRunner):

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(**config.dataset)
        sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=False,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=dataset.collate,
        )
        return dataset, sampler, dataloader

    def _build_model(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        model = Model(config=config).requires_grad_()
        if not debug.CPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in batch])
        logits = self._model(batch)
        memo['results'].append((logits, batch.bbox_labels))

    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        if log:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if not debug.CPU:
            results = tuple(  # all_gather must exec for all ranks
                map(all_gather, results),
            )
        if todd.base.get_rank() == 0:
            results = map(lambda result: torch.cat(result).cpu().numpy(), results)
            logits, labels = results
            raise NotImplementedError
            # self._logger.info(f"mAP {mAP * 100:.3f}")
            # return mAP
        else:
            return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    runner = Runner(name=args.name, config=config)
    runner.run()

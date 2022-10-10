import argparse
from ast import Num
import enum
import math
import os
import pathlib
import pickle
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, cast

import PIL.Image
import clip
import clip.model
import einops
import todd
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms

from .debug import debug
from .todd import BaseRunner, TrainerMixin
from .utils import odps_init, k8s_init

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC


class Batch(NamedTuple):
    image_ids: List[int]
    patches: torch.Tensor
    bboxes: torch.Tensor
    num_patches: List[int]


class CocoClassification(torchvision.datasets.CocoDetection):

    def __init__(
        self,
        root: str,
        ann_file: str,
        proposal_file: str,
        embeddings_root: str,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )
        # self.ids = list(self.coco.imgs.keys())

        with open(proposal_file, 'rb') as f:
            self.proposals = torch.tensor(
                pickle.load(f),
                dtype=torch.float,
            )

        self._embeddings_root = pathlib.Path(embeddings_root)
        self._embeddings_root.mkdir(parents=True, exist_ok=True)

        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    @property
    def embeddings_root(self) -> pathlib.Path:
        return self._embeddings_root

    def _crop(self, image: PIL.Image.Image, bboxes: todd.BBoxesXYXY) -> torch.Tensor:
        return torch.stack([
            self._transform(image.crop(bbox))
            for bbox in bboxes.round().to_tensor().int().tolist()
        ])

    def __getitem__(self, index: int) -> Optional[Tuple[int, torch.Tensor, torch.Tensor]]:
        image_id = self.ids[index]
        embedding_file = self._embeddings_root / f'{image_id:012d}.pth'
        if not debug.DRY_RUN and embedding_file.exists():
            try:
                torch.load(embedding_file, map_location='cpu')
                return None
            except Exception:
                pass
        image = self._load_image(image_id)
        proposals = todd.BBoxesXYXY(self.proposals[index][:, :4])
        proposals = proposals[proposals.indices(min_wh=(32, 32))]
        patches = torch.stack([
            self._crop(image, proposals),
            self._crop(image, proposals.expand(1.5, image_wh=image.size)),
        ])
        return (
            image_id,
            patches,
            proposals.to_tensor()
        )

    @staticmethod
    def collate(batch: List[Optional[Tuple[int, torch.Tensor, torch.Tensor]]]) -> Optional[Batch]:
        batch: List[Tuple[int, torch.Tensor, torch.Tensor]] = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        image_id_list, patches_list, bboxes_list = zip(*batch)
        num_patches = list(map(len, bboxes_list))
        return Batch(image_id_list, torch.cat(patches_list, 1), torch.cat(bboxes_list), num_patches)


class Model(todd.reproduction.FrozenMixin, todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        self._visual = clip_model.visual

        frozen_names = ['_visual']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=frozen_names),
            eval_config=dict(names=frozen_names),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._visual(x)
        return F.normalize(x)


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
        config = config.copy()
        pretrained = config.pop('pretrained')
        assert "ViT-B-32" in pretrained
        if debug.CPU:
            clip_model, _ = clip.load(pretrained, 'cpu')
        else:
            clip_model, _ = clip.load(pretrained)
        clip_model.requires_grad_(False)
        model = Model(clip_model=clip_model, **config)
        if not debug.CPU:
            model = model.cuda()
        return model

    def _run_iter(
        self, *args, i: int, batch: Optional[Batch], memo: Dict[str, Any], **kwargs,
    ) -> None:
        if batch is None:
            return
        patches: torch.Tensor = batch.patches
        if debug.DRY_RUN:
            patches = patches[:, :5]
            batch = Batch(
                batch.image_ids[:2],
                patches,
                batch.bboxes[:5],
                [2, 3],
            )
        if not debug.CPU:
            patches = patches.half().cuda()
        n = patches.shape[0]
        patches = einops.rearrange(patches, 'n b c h w -> (n b) c h w')
        patches = self._model(patches)
        patches = einops.reduce(patches, '(n b) c -> b c', n=n, reduction='mean')
        patches_list = patches.half().split(batch.num_patches)
        bboxes_list = batch.bboxes.half().split(batch.num_patches)
        memo['results'] = [
            dict(bboxes=bboxes, patches=patches)
            for bboxes, patches in zip(bboxes_list, patches_list)
        ]

    def _after_run_iter(self, *args, i: int, batch: Optional[Batch], memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        if batch is not None:
            assert 'results' in memo
            for image_id, result in zip(batch.image_ids, memo.pop('results')):
                torch.save(
                    result,
                    self._dataset.embeddings_root / f'{image_id:012d}.pth',
                )
        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(self, *args, **kwargs) -> Tuple[
        CocoClassification,
        None,
        torch.utils.data.DataLoader,
    ]:
        return self._build_dataloader(*args, **kwargs)

    def _build_train_fixtures(self, *args, **kwargs) -> None:
        pass

    @torch.no_grad()
    def _train_iter(self, *args, **kwargs) -> None:
        return self._run_iter(*args, **kwargs)

    def _after_train_iter(
        self,
        *args,
        epoch: int,
        i: int,
        batch: Optional[Batch],
        memo: Dict[str, Any],
        log: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        if batch is not None:
            assert 'results' in memo
            for image_id, result in zip(batch.image_ids, memo.pop('results')):
                torch.save(
                    result,
                    self._train_dataset.embeddings_root / f'{image_id:012d}.pth',
                )
        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Train Step [{i}/{len(self._train_dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--k8s', action=todd.base.DictAction)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    if args.k8s is not None:
        k8s_init(args.k8s)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        if args.local_rank is not None:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        torch.cuda.set_device(todd.base.get_local_rank())

    todd.reproduction.init_seed(args.seed)

    trainer = Trainer(name=args.name, config=config)
    trainer.train()
    trainer.run()

from collections import namedtuple
import itertools
from typing import Any, Dict, List, Literal, Optional, Tuple

import argparse
import sys
import pathlib
import PIL.Image
import einops

import todd
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.distributed

import clip
import clip.model
from .utils import odps_init

from .debug import debug
from .todd import BaseRunner, TrainerMixin

Batch = namedtuple('Batch', ['image', 'patches', 'bboxes', 'image_id'])


class CocoClassification(torchvision.datasets.coco.CocoDetection):

    def __init__(
        self,
        root: str,
        ann_file: str,
        patch_size: int = 224,
        max_stride: int = 112,
        rescale: float = 1.5,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )

        self._patch_size = patch_size
        self._max_stride = max_stride
        self._rescale = rescale
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _cut(self, length: int) -> List[int]:
        assert length >= self._patch_size

        result = [0]
        if length != self._patch_size:
            n = (length - self._patch_size - 1) // self._max_stride + 1
            q, r = divmod(length - self._patch_size, n)
            for i in range(n):
                result.append(result[-1] + q + (i < r))
        return result

    def __getitem__(
        self,
        index: int,
    ) -> Batch:
        image_id = self.ids[index]
        image = self._load_image(image_id)
        image_ = self._transform(image)

        scale = 1
        patches = []
        bboxes = []
        while image.width >= self._patch_size and image.height >= self._patch_size:
            for x, y in itertools.product(*map(self._cut, image.size)):
                patch = image\
                    .crop((x, y, x + self._patch_size, y + self._patch_size))\
                    .resize((224, 224), PIL.Image.Resampling.BICUBIC)
                bbox = (x, y, self._patch_size, scale)
                patches.append(patch)
                bboxes.append(bbox)
            image = image.resize((int(image.width / self._rescale), int(image.height / self._rescale)))
            scale *= self._rescale

        if len(patches) == 0:
            patches_ = torch.empty(0, 3, 224, 224)
            bboxes_ = torch.empty(0, 4)
        else:
            patches_ = torch.stack(list(map(self._transform, patches)))
            bboxes_ = torch.tensor(bboxes)
            bboxes_[:, :-1] *= bboxes_[:, [-1]]
            bboxes_[:, -1] = bboxes_[:, -2]

        return Batch(image_, patches_, bboxes_, image_id)

    @staticmethod
    def collate(
        batch: List[Batch],
    ) -> Batch:
        assert len(batch) == 1
        batch[0].image.unsqueeze_(0),
        return batch[0]


class ImageEncoder(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        visual = clip_model.visual
        self._conv1 = visual.conv1
        self._ln_pre = visual.ln_pre
        self._transformer = visual.transformer
        self._ln_post = visual.ln_post
        self._proj = visual.proj

        spatial_size = visual.input_resolution // visual.conv1.kernel_size[0]
        assert visual.positional_embedding.shape[0] == spatial_size ** 2 + 1
        ce = visual.class_embedding + visual.positional_embedding[0]
        pe = einops.rearrange(
            visual.positional_embedding[1:],
            '(h w) c -> 1 c h w',
            h=spatial_size,
            w=spatial_size,
        )
        self._ce = nn.Parameter(ce)
        self._pe = nn.Parameter(pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._conv1(x)
        b, _, h, w = x.shape

        pe = self._pe
        if pe.shape[-2:] != (h, w):
            pe = F.interpolate(pe, size=(h, w), mode='bilinear')
        x += pe
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        ce = einops.repeat(self._ce, 'c -> b 1 c', b=b)
        x = torch.cat([ce, x], dim=1)

        x = self._ln_pre(x)
        x = einops.rearrange(x, 'b l c -> l b c')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l b c -> b l c')
        x = self._ln_post(x[:, 0, :])
        x = x @ self._proj
        return F.normalize(x)


class Runner(BaseRunner):

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        None,
        torch.utils.data.DataLoader,
    ]:
        dataset = CocoClassification(**config.dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=config.num_workers,
            collate_fn=dataset.collate,
        )
        return dataset, None, dataloader,

    def _build_model(
        self,
        *args,
        config: todd.base.Config,
        **kwargs,
    ) -> nn.Module:
        clip_model, _ = clip.load(config.pretrained, 'cpu')
        clip_model.requires_grad_(False)
        model = ImageEncoder(clip_model=clip_model)
        if not debug.CPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        embeddings_root = pathlib.Path(self._config.embeddings_root)
        train_root = embeddings_root / 'train'
        val_root = embeddings_root / 'val'
        train_root.mkdir(parents=True, exist_ok=True)
        val_root.mkdir(parents=True, exist_ok=True)
        memo['train_root'] = train_root
        memo['val_root'] = val_root
        return memo

    def _run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs) -> None:
        image = batch.image
        patches = batch.patches
        if not debug.CPU:
            image = image.cuda()
            patches = patches.cuda()
        image_feature = self._model(image)
        patch_features = (
            self._model(patches) if patches.numel() > 0 else
            torch.empty(0, image_feature.shape[1])
        )
        memo['result'] = dict(
            image=image_feature.clone(),
            patches=patch_features.clone(),
            bboxes=batch.bboxes,
        )

    def _after_run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs) -> None:
        torch.save(
            memo.pop('result'),
            memo['val_root'] / f'{batch.image_id:012d}.pth',
        )
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(self, *args, **kwargs) -> Tuple[
        CocoClassification,
        None,
        torch.utils.data.DataLoader,
    ]:
        return self._build_dataloader(*args, **kwargs)

    def _build_train_fixtures(self, *args, **kwargs) -> None:
        pass

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        return self._before_run(*args, **kwargs)

    def _train_iter(self, *args, **kwargs) -> None:
        return self._run_iter(*args, **kwargs)

    def _after_train_iter(
        self,
        *args,
        epoch: int,
        i: int,
        batch: Batch,
        memo: Dict[str, Any],
        **kwargs,
    ) -> None:
        torch.save(
            memo.pop('result'),
            memo['train_root'] / f'{batch.image_id:012d}.pth',
        )
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Train Step [{i}/{len(self._train_dataloader)}] '
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--seed', type=int, default=3407)
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

    todd.reproduction.init_seed(args.seed)

    trainer = Trainer(name=args.name, config=config)
    trainer.train()
    trainer.run()

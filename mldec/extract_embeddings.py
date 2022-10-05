import argparse
import itertools
import pathlib
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import PIL.Image
import clip
import clip.model
import einops
import todd
import torch
import torch.cuda
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

Batch = namedtuple('Batch', ['image', 'image_id', 'patches', 'bboxes'])


class CocoClassification(torchvision.datasets.CocoDetection):

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
        self._resize_crop = transforms.Compose([
            transforms.Resize(224, interpolation=PIL.Image.Resampling.BICUBIC),
            transforms.CenterCrop(224),
        ])
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
        image_ = self._transform(image).unsqueeze(0)

        scale = 1
        patches = [image]
        bboxes = [(
            (image.width - min(image.size)) / 2,
            (image.height - min(image.size)) / 2,
            min(image.size),
            scale,
        )]
        while image.width >= self._patch_size and image.height >= self._patch_size:
            for x, y in itertools.product(*map(self._cut, image.size)):
                patch = image.crop((x, y, x + self._patch_size, y + self._patch_size))
                bbox = (x, y, self._patch_size, scale)
                patches.append(patch)
                bboxes.append(bbox)
            image = image.resize((int(image.width / self._rescale), int(image.height / self._rescale)))
            scale *= self._rescale
        patches = map(self._resize_crop, patches)
        patches = map(self._transform, patches)
        patches = list(patches)
        patches_ = torch.stack(patches)
        bboxes_ = torch.tensor(bboxes)
        bboxes_[:, :-1] *= bboxes_[:, [-1]]
        bboxes_[:, -1] = bboxes_[:, -2]

        return Batch(image_, image_id, patches_, bboxes_)

    @staticmethod
    def collate(batch: List[Batch]) -> Batch:
        assert len(batch) == 1
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
        return x


class Model(todd.reproduction.FrozenMixin, todd.base.Module):

    def __init__(self, *args, clip_model: clip.model.CLIP, **kwargs) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        self._visual = clip_model.visual
        self._interpolated_visual = ImageEncoder(clip_model=clip_model)
        frozen_names = ['_visual', '_interpolated_visual']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=frozen_names),
            eval_config=dict(names=frozen_names),
        )

    def forward(self, image: torch.Tensor, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._interpolated_visual(image)
        patches = self._visual(patches)
        return F.normalize(image), F.normalize(patches)


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
        clip_model, _ = clip.load(config.pretrained, 'cpu')
        clip_model.requires_grad_(False)
        model = Model(clip_model=clip_model)
        if not debug.CPU:
            model = model.cuda()
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        embeddings_root = pathlib.Path(self._config.embeddings_root) / 'val'
        embeddings_root.mkdir(parents=True, exist_ok=True)
        memo['embeddings_root'] = embeddings_root
        return memo

    def _run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs) -> None:
        image = batch.image
        patches = batch.patches
        if not debug.CPU:
            image = image.cuda()
            patches = patches.cuda()
        image, patches = self._model(image, patches)
        memo['result'] = dict(
            image=image.half(),
            patches=patches.half(),
            bboxes=batch.bboxes.half(),
        )

    def _after_run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        torch.save(
            memo.pop('result'),
            memo['embeddings_root'] / f'{batch.image_id:012d}.pth',
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

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = self._before_run(*args, **kwargs)
        embeddings_root = memo['embeddings_root'].parent / 'train'
        embeddings_root.mkdir(parents=True, exist_ok=True)
        memo['train_embeddings_root'] = embeddings_root
        return memo

    @torch.no_grad()
    def _train_iter(self, *args, **kwargs) -> None:
        return self._run_iter(*args, **kwargs)

    def _after_train_iter(
        self,
        *args,
        epoch: int,
        i: int,
        batch: Batch,
        memo: Dict[str, Any],
        log: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        torch.save(
            memo.pop('result'),
            memo['train_embeddings_root'] / f'{batch.image_id:012d}.pth',
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
    parser.add_argument('--local_rank', type=int, default=0)
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
        torch.cuda.set_device(todd.base.get_local_rank())

    todd.reproduction.init_seed(args.seed)

    trainer = Trainer(name=args.name, config=config)
    trainer.train()
    trainer.run()

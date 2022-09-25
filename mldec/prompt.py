from abc import ABC, abstractmethod
import argparse
from collections import namedtuple
import logging
import pathlib
import re
import time
from typing import Any, Dict, Generator, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union
import math
import sklearn.metrics
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
# import torch.nn.modules.module

import clip
import clip.model
import einops

import todd
import torchvision

from . import losses
from . import datasets

from .debug import debug
from .utils import all_gather, odps_init
from .todd import BaseRunner, TrainerMixin

Batch = namedtuple(
    'Batch',
    ['images', 'class_embeddings', 'class_lengths', 'labels'],
)


class CocoClassification(torchvision.datasets.CocoDetection):
    _classnames: List[str]
    _cat2label: Dict[int, int]

    def __init__(
        self,
        root: str,
        ann_file: str,
        clip_model: clip.model.CLIP,
        split: Optional[str] = None,
        filter_empty: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ])
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

        if filter_empty:
            self.ids = list(filter(self._load_target, self.ids))

        class_tokens = clip.tokenize([
            name.replace('_', ' ') + '.'
            for name in self._classnames
        ])
        with torch.no_grad():
            self._class_embeddings = clip_model.token_embedding(class_tokens)
        self._class_lengths = class_tokens.argmax(dim=-1)

    @property
    def classnames(self) -> Tuple[str]:
        return tuple(self._classnames)

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [anno for anno in target if anno['category_id'] in self._cat2label]

    def _load_labels(self, target: List[Any]) -> torch.Tensor:
        bbox_labels = [
            self._cat2label[anno['category_id']]
            for anno in target
        ]
        labels = torch.zeros(len(self._classnames), dtype=torch.bool)
        labels[bbox_labels] = True
        return labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image, target = super().__getitem__(index)
        labels = self._load_labels(target)
        return image, labels

    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Batch:
        images, labels = map(
            torch.utils.data.dataloader.default_collate,
            zip(*batch),
        )
        return Batch(images, self._class_embeddings, self._class_lengths, labels)


class TextPrompt(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        prompt: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompt_tokens = clip.tokenize([prompt])[0, 1:-1]
        with torch.no_grad():
            prompt_embedding = clip_model.token_embedding(prompt_tokens)
        self._prompt_embedding = nn.Parameter(prompt_embedding)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeddings = einops.repeat(
            self._prompt_embedding,
            'l d -> n l d',
            n=batch.class_embeddings.shape[0],
        )
        x = torch.cat(
            [
                batch.class_embeddings[:, :1],
                prompt_embeddings,
                batch.class_embeddings[:, 1:],
            ],
            dim=1,
        )
        l = batch.class_lengths + self._prompt_embedding.shape[0]
        return x, l


class TextEncoder(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = clip_model.transformer
        self._pe = clip_model.positional_embedding
        self._ln = clip_model.ln_final
        self._proj = clip_model.text_projection

    def forward(self, batch: Batch, prompt: TextPrompt) -> torch.Tensor:
        x, l = prompt(batch)
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self._proj
        return F.normalize(x)


class ImagePrompt(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        length: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        width = clip_model.visual.conv1.out_channels
        scale = width ** -0.5
        self._prompt = nn.Parameter(scale * torch.randn((length, width)))

    def forward(self, batch: Batch) -> torch.Tensor:
        return einops.repeat(self._prompt, 'l d -> n l d', n=batch.images.shape[0])


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
        self._ce = visual.class_embedding + visual.positional_embedding[0]
        self._pe = einops.rearrange(
            visual.positional_embedding[1:],
            '(h w) c -> 1 c h w',
            h=spatial_size,
            w=spatial_size,
        )

    def forward(self, batch: Batch, prompt: ImagePrompt) -> torch.Tensor:
        x = self._conv1(batch.images)
        b, _, h, w = x.shape
        ce = einops.repeat(self._ce, 'c -> b 1 c', b=b)
        x += F.interpolate(self._pe, size=(h, w), mode='bilinear')
        x = einops.rearrange(x, 'b c h w -> b (h w) c')
        prompt_ = prompt(batch)
        x = torch.cat([ce, x, prompt_], dim=1)
        x = self._ln_pre(x)
        x = einops.rearrange(x, 'b l c -> l b c')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l b c -> b l c')
        x = self._ln_post(x[:, 0, :])
        x = x @ self._proj
        return F.normalize(x)


class Model(todd.reproduction.FrozenMixin, todd.base.Module):

    def _is_frozen(self, key: str) -> bool:
        return any(name in key for name in self._frozen_names)

    @staticmethod
    def _state_dict_hook(
        self: 'Model',
        destination: MutableMapping[str, Any],
        prefix,
        local_metadata,
    ) -> None:
        for k in list(filter(self._is_frozen, destination.keys())):
            destination.pop(k)

    @staticmethod
    def _load_state_dict_post_hook(self: 'Model', keys: nn.modules.module._IncompatibleKeys) -> None:
        missing_keys = keys.missing_keys

        i = 0
        while i < len(missing_keys):
            if self._is_frozen(missing_keys[i]):
                missing_keys.pop(i)
            else:
                i += 1

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        config: todd.base.Config,
        **kwargs
    ) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        self._text_prompt = TextPrompt(
            clip_model=clip_model,
            **config.text_prompt,
        )
        self._text_encoder = TextEncoder(
            clip_model=clip_model,
            **config.text_encoder,
        )
        self._image_prompt = ImagePrompt(
            clip_model=clip_model,
            **config.image_prompt,
        )
        self._image_encoder = ImageEncoder(
            clip_model=clip_model,
            **config.image_encoder,
        )
        self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

        self._frozen_names = ['_text_encoder', '_image_encoder']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=self._frozen_names),
            eval_config=dict(names=self._frozen_names),
        )

        self._register_state_dict_hook(self._state_dict_hook)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def forward(self, batch: Batch) -> torch.Tensor:
        texts = self.encode_texts(batch)
        images = self.encode_images(batch)
        return (images @ texts.T) * self._scaler - self._bias

    def encode_images(self, batch: Batch) -> torch.Tensor:
        return self._image_encoder(batch, self._image_prompt)

    def encode_texts(self, batch: Batch) -> torch.Tensor:
        return self._text_encoder(batch, self._text_prompt)

    def dump_texts(self, batch: Batch) -> Dict[str, torch.Tensor]:
        return dict(
            embeddings=self.encode_texts(batch),
            scaler=self._scaler,
            bias=self._bias,
        )


class Runner(BaseRunner):

    def __init__(self, *args, **kwargs) -> None:
        clip_model, _ = clip.load('pretrained/clip/ViT-B-32.pt', 'cpu')
        super().__init__(*args, clip_model=clip_model, **kwargs)

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(clip_model=clip_model, **config.dataset)
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
            num_workers=config.workers,
            collate_fn=dataset.collate,
        )
        return dataset, sampler, dataloader

    def _build_model(
        self,
        *args,
        config: Optional[todd.base.Config],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        # `requires_grad_` must be called on `Model` and nowhere else
        model = Model(
            clip_model=clip_model,
            config=config,
        ).requires_grad_()
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
            batch = Batch(*[x.cuda() for x in batch])
        logits = self._model(batch)
        memo['results'].append((logits, batch.labels))

    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if not debug.CPU:
            results = tuple(  # all_gather must exec for all ranks
                map(all_gather, results),
            )
        if todd.base.get_rank() == 0:
            results = map(lambda result: torch.cat(result).cpu().numpy(), results)
            image_logits, image_labels = results
            mAP = sklearn.metrics.average_precision_score(image_labels, image_logits)
            self._logger.info(f"mAP: {mAP * 100:.3f}")
            return mAP
        else:
            return -1

    def dump(self) -> None:
        model = self._model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        state_dict = model.dump_texts(self._dataset[0])
        state_dict.update(
            names=self._dataset.classnames,
        ),
        torch.save(state_dict, self._work_dir / f'epoch_{self._epoch}_classes.pth')


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(clip_model=clip_model, **config.dataset)
        sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=not debug.CPU,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.workers,
            collate_fn=dataset.collate,
        )
        return dataset, sampler, dataloader

    def _build_train_fixtures(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> None:
        assert config is not None
        self._criterion = todd.losses.LOSSES.build(config.loss)

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_train(self, *args, **kwargs)
        memo.update(
            record=-1,
        )
        return memo

    def _train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        outputs = self._model(batch)
        loss: torch.Tensor = self._criterion(outputs.sigmoid(), batch.labels)
        self._model.zero_grad()
        loss.backward()
        self._optimizer.step()
        memo['loss'] = loss.item()

    def _after_train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Epoch [{epoch}/{self._config.train.epoch}] '
                f'Train Step [{i}/{len(self._train_dataloader)}] '
                # f'LR {self._scheduler.get_last_lr()[0]:.3e} '
                f'Loss {memo.pop("loss"):.3f}'
            )

    def _after_train_epoch(self, *args, epoch: int, memo: Dict[str, Any], **kwargs) -> None:
        # self._scheduler.step()
        if todd.base.get_rank() == 0:
            self.save_checkpoint(epoch)
        mAP = self.run()
        memo.update(
            record=max(memo['record'], mAP),
        )
        if todd.base.get_rank() == 0:
            self._logger.info(
                f"record: {memo['record'] * 100:.3f}, "
            )

    def _after_train(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        return memo['record']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', choices=['train', 'val', 'dump'])
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--load', type=int)
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

    init_dict = dict(name=args.name, config=config, load=args.load)
    if args.mode == 'train':
        runner = Trainer(**init_dict)
    else:
        runner = Runner(**init_dict)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'val':
        runner.run()
    else:
        runner.dump()

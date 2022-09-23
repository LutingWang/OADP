from abc import ABC, abstractmethod
import argparse
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
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
# import torch.nn.modules.module

import clip
import clip.model
import einops

import todd

from . import losses

from .datasets import Batch, CocoClassification
from .debug import debug
from .utils import all_gather, odps_init
from .todd import BaseRunner, TrainerMixin


class Classnames(todd.base.Module):

    def __init__(
        self,
        *args,
        classnames: Sequence[str],
        embedding: nn.Embedding,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        classnames = [classname.replace('_', ' ') + '.' for classname in classnames]
        classname_tokens = clip.tokenize(classnames)
        self._lengths = classname_tokens.argmax(dim=-1)
        with torch.no_grad():
            classname_embeddings = embedding(classname_tokens)
        self.register_buffer('_embeddings', classname_embeddings, persistent=False)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    @property
    def embeddings(self) -> torch.Tensor:
        return self.get_buffer('_embeddings')

    @property
    def lengths(self) -> torch.Tensor:
        return self._lengths


class Prompt(todd.base.Module):

    def __init__(
        self,
        *args,
        prompt: str,
        embedding: nn.Embedding,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompt_tokens = clip.tokenize([prompt])[0, 1:-1]
        with torch.no_grad():
            prompt_embedding: torch.Tensor = embedding(prompt_tokens)
        self._prompt_embedding = nn.Parameter(prompt_embedding)

    def __len__(self) -> int:
        return self._prompt_embedding.shape[0]

    def forward(
        self,
        classnames: Classnames,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embedding = einops.repeat(
            self._prompt_embedding,
            'l d -> n l d',
            n=len(classnames),
        )
        embeddings = torch.cat(
            [
                classnames.embeddings[:, :1],
                prompt_embedding,
                classnames.embeddings[:, 1:],
            ],
            dim=1,
        )
        lengths = classnames.lengths + len(self)
        return embeddings, lengths


class CLIPTextEncoder(todd.base.Module):

    def __init__(self, *args, clip_model: clip.model.CLIP, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = clip_model.transformer
        self._pe = clip_model.positional_embedding
        self._ln = clip_model.ln_final
        self._proj = clip_model.text_projection

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self._proj
        return x


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
        self._clip_text_encoder = CLIPTextEncoder(clip_model=clip_model)
        self._prompt = Prompt(
            prompt=config.prompt,
            embedding=clip_model.token_embedding.cpu(),
        )
        self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

        self._frozen_names = ['_clip_text_encoder']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=self._frozen_names),
            eval_config=dict(names=self._frozen_names),
        )

        self._register_state_dict_hook(self._state_dict_hook)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def get_classes(self, classes: Classnames) -> torch.Tensor:
        classes, l = self._prompt(classes)
        classes = self._clip_text_encoder.forward(classes, l)
        return F.normalize(classes)

    def forward(
        self,
        batch: Batch,
        classes: Union[Classnames, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(classes, Classnames):
            classes = self.get_classes(classes)
        return (batch.patches @ classes.T) * self._scaler - self._bias, classes

    def dump(self, classes: Classnames) -> Dict[str, torch.Tensor]:
        return dict(
            embeddings=self.get_classes(classes),
            scaler=self._scaler,
            bias=self._bias,
        )


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
            num_workers=config.workers,
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
        clip_model, _ = clip.load('pretrained/clip/RN50.pt', 'cpu')
        self._token_embedding = clip_model.token_embedding

        classnames = Classnames(
            classnames=self._dataset.classnames,
            embedding=self._token_embedding,
        )
        # `requires_grad_` must be called on `Model` and nowhere else
        model = Model(
            clip_model=clip_model,
            config=config,
        ).requires_grad_()
        if not debug.CPU:
            classnames = classnames.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        self._classnames = classnames
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(classes=self._classnames, results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        patch_logits, classes = self._model(batch, memo['classes'])
        image_logits = torch.stack(
            list(
                map(
                    lambda x: x.max(0).values,
                    patch_logits.split(batch.num_patches.tolist()),
                ),
            ),
        )
        memo['classes'] = classes
        memo['results'].append((image_logits, batch.image_labels, patch_logits, batch.patch_labels))

    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> Tuple[float, float]:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if not debug.CPU:
            results = tuple(  # all_gather must exec
                map(all_gather, results),
            )
        if todd.base.get_rank() == 0:
            results = map(lambda result: torch.cat(result).cpu().numpy(), results)
            image_logits, image_labels, patch_logits, patch_labels = results
            image_mAP = sklearn.metrics.average_precision_score(image_labels, image_logits)
            patch_mAP = sklearn.metrics.average_precision_score(patch_labels, patch_logits)
            self._logger.info(f"image_mAP: {image_mAP * 100:.3f}, patch_mAP: {patch_mAP * 100:.3f}")
            return image_mAP, patch_mAP
        else:
            return -1, -1

    def dump(self) -> None:
        model = self._model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        classnames = [cat['name'] for cat in self._dataset.coco.cats.values()]
        state_dict = model.dump(
            Classnames(
                classnames=classnames,
                embedding=self._token_embedding,
            ),
        )
        state_dict.update(
            classnames=classnames,
        ),
        torch.save(state_dict, self._work_dir / f'epoch_{self._epoch}_embeddings.pth')


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(
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
        classnames = Classnames(
            classnames=self._train_dataset.classnames,
            embedding=self._token_embedding,
        )
        if not debug.CPU:
            classnames = classnames.cuda()
        self._train_classnames = classnames

        self._criterion = todd.losses.LOSSES.build(config.loss)

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_train(self, *args, **kwargs)
        memo.update(
            image_record=-1,
            patch_record=-1,
        )
        return memo

    def _train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        outputs, _ = self._model(batch, self._train_classnames)
        loss: torch.Tensor = self._criterion(outputs.sigmoid(), batch.patch_labels)
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
        image_mAP, patch_mAP = self.run()
        memo.update(
            image_record=max(memo['image_record'], image_mAP),
            patch_record=max(memo['patch_record'], patch_mAP),
        )
        if todd.base.get_rank() == 0:
            self._logger.info(
                f"image_record: {memo['image_record'] * 100:.3f}, "
                f"patch_record: {memo['patch_record'] * 100:.3f}"
            )

    def _after_train(self, *args, memo: Dict[str, Any], **kwargs) -> Tuple[float, float]:
        return memo['image_record'], memo['patch_record']


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

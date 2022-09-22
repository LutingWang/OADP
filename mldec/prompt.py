import argparse
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


IMAGE_REFINERS = todd.base.Registry('image refiners')


@IMAGE_REFINERS.register_module(name='Base')
class BaseImageRefiner(todd.base.Module):

    def forward(self, batch: Batch, classes: torch.Tensor) -> torch.Tensor:
        return batch.patches


@IMAGE_REFINERS.register_module(name='TransformerEncoderLayer')
class TransformerEncoderLayerImageRefiner(todd.base.Module):

    def __init__(self, *args, num_channels: int, num_heads: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pe = nn.Linear(4, num_channels, bias=False)
        self._transformer = nn.TransformerEncoderLayer(num_channels, num_heads, batch_first=True)

    def forward(self, batch: Batch, classes: torch.Tensor) -> torch.Tensor:
        batch_size = batch.num_patches.numel()
        length = batch.num_patches.max().item()
        dim = batch.patches.shape[-1]
        images = batch.patches.new_zeros(batch_size, length, dim)
        masks = batch.patches.new_ones((batch_size, length), dtype=bool)

        patches: torch.Tensor = batch.patches + self._pe(batch.patch_bboxes)
        patches_list = patches.split(batch.num_patches.tolist())
        for i, patches in enumerate(patches_list):
            images[i, :batch.num_patches[i]] = patches
            masks[i, :batch.num_patches[i]] = False
        images_ = self._transformer(images, src_key_padding_mask=masks)
        images_ = torch.cat([
            images_[i, :num_patches] for i, num_patches in enumerate(batch.num_patches.tolist())
        ])

        return images_


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
        self._image_refiner = IMAGE_REFINERS.build(config.image_refiner)
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

    def forward(
        self,
        batch: Batch,
        classes: Union[Classnames, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(classes, Classnames):
            classes, l = self._prompt(classes)
            classes = self._clip_text_encoder.forward(classes, l)
            classes = F.normalize(classes)
        images = self._image_refiner(batch, classes)
        return (images @ classes.T) * self._scaler - self._bias, classes


class Runner:

    def __init__(self, name: str, config: todd.base.Config, load: Optional[int]) -> None:
        self._name = name
        self._config = config

        self._build_work_dir()
        self._build_logger()
        self._build_dataloader()

        clip_model, _ = clip.load('pretrained/clip/RN50.pt', 'cpu')
        self._token_embedding = clip_model.token_embedding

        classnames = Classnames(
            classnames=self._dataset.classnames,
            embedding=self._token_embedding,
        )
        # `requires_grad_` must be called on `Model` and nowhere else
        model = Model(
            clip_model=clip_model,
            config=config.model,
        ).requires_grad_()
        if not debug.CPU:
            classnames = classnames.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        self._classnames = classnames
        self._model = model

        if load is not None:
            self.load_checkpoint(load)

    def _build_logger(self):
        if todd.base.get_rank() == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = self._work_dir / f'{timestamp}.log'
            todd.base.init_log_file(log_file)
        self._logger = todd.base.get_logger()
        self._logger.info(f"Version {todd.base.git_commit_id()}")
        self._logger.info(f"Config\n{self._config.dumps()}")

    def _build_work_dir(self) -> None:
        self._work_dir = pathlib.Path(f'work_dirs/{self._name}')
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataloader(self) -> None:
        config = self._config.val
        self._dataset = CocoClassification(**config.dataset)
        self._sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                self._dataset,
                shuffle=False,
            )
        )
        self._dataloader = torch.utils.data.DataLoader(
            self._dataset,
            batch_size=config.batch_size,
            sampler=self._sampler,
            num_workers=config.workers,
            collate_fn=self._dataset.collate,
        )

    def load_checkpoint(self, epoch: int) -> None:
        todd.base.load_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
        )

    @torch.no_grad()
    def run(self) -> Tuple[float, float]:
        self._model.eval()
        classes = self._classnames
        results = []
        for i, batch in enumerate(self._dataloader):
            *result, classes = self.run_iter(batch, classes)
            results.append(result)
            if i % self._config.log_interval != 0:
                continue
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
            if debug.LESS_DATA and i: break
        results_iter: Iterator[Tuple[torch.Tensor, ...]] = zip(*results)
        if not debug.CPU:
            results_iter = tuple(  # all_gather must exec
                map(all_gather, results_iter),
            )
        if todd.base.get_rank() == 0:
            results_iter = map(lambda results: torch.cat(results).cpu().numpy(), results_iter)
            image_logits, image_labels, patch_logits, patch_labels = results_iter
            image_mAP = sklearn.metrics.average_precision_score(image_labels, image_logits)
            patch_mAP = sklearn.metrics.average_precision_score(patch_labels, patch_logits)
            self._logger.info(f"image_mAP: {image_mAP * 100:.3f}, patch_mAP: {patch_mAP * 100:.3f}")
            return image_mAP, patch_mAP
        else:
            return -1, -1

    def run_iter(
        self, batch: Batch, classes: Union[Classnames, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        patch_logits, classes = self._model(batch, classes)
        image_logits = torch.stack(
            list(
                map(
                    lambda x: x.max(0).values,
                    patch_logits.split(batch.num_patches.tolist()),
                ),
            ),
        )
        return image_logits, batch.image_labels, patch_logits, batch.patch_labels, classes


class Trainer(Runner):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        config = self._config.train
        self._train_dataset = CocoClassification(**config.dataset)
        self._train_sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                self._train_dataset,
                shuffle=not debug.CPU,
            )
        )
        self._train_dataloader = torch.utils.data.DataLoader(
            self._train_dataset,
            batch_size=config.batch_size,
            sampler=self._train_sampler,
            num_workers=config.workers,
            collate_fn=self._train_dataset.collate,
        )

        classnames = Classnames(
            classnames=self._train_dataset.classnames,
            embedding=self._token_embedding,
        )
        if not debug.CPU:
            classnames = classnames.cuda()
        self._train_classnames = classnames

        self._criterion = todd.losses.LOSSES.build(config.loss)
        self._optimizer = todd.utils.OPTIMIZERS.build(
            config.optimizer,
            default_args=dict(model=self._model),
        )
        self._scheduler = todd.utils.LR_SCHEDULERS.build(
            config.lr_scheduler,
            default_args=dict(optimizer=self._optimizer),
        )

        self._start_epoch = 0

    def load_checkpoint(self, epoch: int) -> None:
        todd.base.load_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
            optimizer=self._optimizer,
            scheduler=self._scheduler,
        )
        self._start_epoch = epoch + 1

    def save_checkpoint(self, epoch: int) -> None:
        todd.base.save_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
            optimizer=self._optimizer, scheduler=self._scheduler,
        )

    def train(self) -> Tuple[float, float]:
        image_record = -1
        patch_record = -1
        for epoch in range(self._start_epoch, self._config.train.epoch):
            if not debug.CPU:
                torch.distributed.barrier()
                self._train_sampler.set_epoch(epoch)
            self.train_epoch(epoch)
            image_mAP, patch_mAP = self.run()
            image_record = max(image_record, image_mAP)
            patch_record = max(patch_record, patch_mAP)
            if todd.base.get_rank() == 0:
                self._logger.info(
                    f"image_record: {image_record * 100:.3f}, "
                    f"patch_record: {patch_record * 100:.3f}"
                )
        return image_record, patch_record

    def train_epoch(self, epoch: int) -> None:
        self._model.train()

        for i, batch in enumerate(self._train_dataloader):
            loss = self.train_iter(batch)
            if i % self._config.log_interval != 0:
                continue
            self._logger.info(
                f'Epoch [{epoch}/{self._config.train.epoch}] '
                f'Train Step [{i}/{len(self._train_dataloader)}] '
                f'LR {self._scheduler.get_last_lr()[0]:.3e} '
                f'Loss {loss:.3f}'
            )
            if debug.LESS_DATA and i: break

        self._scheduler.step()
        if todd.base.get_rank() == 0:
            self.save_checkpoint(epoch)

    def train_iter(self, batch: Batch) -> float:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        outputs, _ = self._model(batch, self._train_classnames)
        loss: torch.Tensor = self._criterion(outputs.sigmoid(), batch.patch_labels)
        self._model.zero_grad()
        loss.backward()
        self._optimizer.step()
        return loss.item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', choices=['train', 'val'])
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

    if args.mode == 'train':
        runner = Trainer(args.name, config, args.load)
        runner.train()
    else:
        runner = Runner(args.name, config, args.load)
        runner.run()

import argparse
import pathlib
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
import todd
import sklearn.metrics

from .utils import all_gather, odps_init

from .debug import debug
from . import losses

from .datasets import Batch, CocoClassification
from .todd import BaseRunner, TrainerMixin


class Classes(todd.base.Module):

    def __init__(self, *args, embeddings: str, names: Sequence[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        state_dict = torch.load(embeddings, 'cpu')
        embeddings = state_dict['embeddings'].requires_grad_(False)
        name_dict = {
            name: i for i, name in enumerate(state_dict['classnames'])
        }
        name_inds = [name_dict[name] for name in names]
        self.embeddings = nn.Parameter(embeddings[name_inds], requires_grad=False)
        self.scaler = state_dict['scaler'].item()
        self.bias = state_dict['bias'].item()


class Model(todd.base.Module):

    def __init__(self, *args, num_channels: int, num_heads: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pe = nn.Linear(4, num_channels, bias=False)
        self._transformer = nn.TransformerEncoderLayer(num_channels, num_heads, batch_first=True)

    def forward(self, batch: Batch, classes: Classes) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch.num_patches.numel()
        length = batch.num_patches.max().item()
        dim = batch.patches.shape[-1]
        inputs = batch.patches.new_zeros(batch_size, length, dim)
        masks = batch.patches.new_ones((batch_size, length), dtype=bool)

        patches: torch.Tensor = batch.patches + self._pe(batch.patch_bboxes)
        patches_list = patches.split(batch.num_patches.tolist())
        for i, patches in enumerate(patches_list):
            inputs[i, :batch.num_patches[i]] = patches
            masks[i, :batch.num_patches[i]] = False
        patches = self._transformer(inputs, src_key_padding_mask=masks)
        images = patches[:, 0]
        patches = torch.cat([
            patches[i, :num_patches] for i, num_patches in enumerate(batch.num_patches.tolist())
        ])

        # return (images_ @ classes.embeddings.T) * classes.scaler - classes.bias
        return (images @ classes.embeddings.T) * classes.scaler - classes.bias, patches


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
        classes = Classes(
            embeddings=self._config.val.class_embeddings,
            names=self._dataset.classnames,
        ).requires_grad_(False)
        model = Model(**config).requires_grad_()
        if not debug.CPU:
            classes = classes.cuda()
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        self._classes = classes
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(classes=self._classes, results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        image_logits, _ = self._model(batch, memo['classes'])
        memo['results'].append((image_logits, batch.image_labels))

    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if not debug.CPU:
            results = tuple(  # all_gather must exec
                map(all_gather, results),
            )
        if todd.base.get_rank() == 0:
            results = map(lambda result: torch.cat(result).cpu().numpy(), results)
            image_logits, image_labels = results
            image_mAP = sklearn.metrics.average_precision_score(image_labels, image_logits)
            self._logger.info(f"image_mAP: {image_mAP * 100:.3f}")
            return image_mAP
        else:
            return -1

    def dump(self) -> None:
        raise NotImplementedError


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
        classes = Classes(
            embeddings=config.class_embeddings,
            names=self._train_dataset.classnames,
        ).requires_grad_(False)
        if not debug.CPU:
            classes = classes.cuda()
        self._train_classes = classes

        self._criterions = todd.base.ModuleDict({
            k: todd.losses.LOSSES.build(v) for k, v in config.losses.items()
        })

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_train(self, *args, **kwargs)
        memo.update(
            image_record=-1,
        )
        return memo

    def _train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        outputs, patches = self._model(batch, self._train_classes)
        ml_loss = self._criterions['ml_loss'](outputs.sigmoid(), batch.image_labels)
        rec_loss = self._criterions['rec_loss'](patches, batch.patches)
        loss = ml_loss + rec_loss
        self._model.zero_grad()
        loss.backward()
        self._optimizer.step()
        memo['losses'] = dict(ml_loss=ml_loss.item(), rec_loss=rec_loss.item())

        todd.base.inc_iter()

    def _after_train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        if i % self._config.log_interval == 0:
            self._logger.info(
                f'Epoch [{epoch}/{self._config.train.epoch}] '
                f'Train Step [{i}/{len(self._train_dataloader)}] '
                f'LR {self._scheduler.get_last_lr()[0]:.3e} '
                f'Loss {memo.pop("losses")}'
            )

    def _after_train_epoch(self, *args, epoch: int, memo: Dict[str, Any], **kwargs) -> None:
        self._scheduler.step()
        if todd.base.get_rank() == 0:
            self.save_checkpoint(epoch)
        image_mAP = self.run()
        memo.update(
            image_record=max(memo['image_record'], image_mAP),
        )
        if todd.base.get_rank() == 0:
            self._logger.info(
                f"image_record: {memo['image_record'] * 100:.3f}, "
            )

    def _after_train(self, *args, memo: Dict[str, Any], **kwargs) -> Tuple[float, float]:
        return memo['image_record']


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

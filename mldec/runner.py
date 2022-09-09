import sklearn.metrics
import functools
import os
import sys
import pathlib
import time
from typing import Optional, Tuple

import torch
import torch.distributed
import torch.nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.optim.lr_scheduler
from randaugment import RandAugment
import torchvision.transforms as tf

import todd

sys.path.insert(0, '')
import clip
import clip.model
import mldec.datasets
from mldec.losses import AsymmetricLoss
from mldec.debug import debug
import mldec.coop


def odps_init(kwargs: todd.base.Config) -> None:
    logger = todd.base.get_logger()
    logger.debug("ODPS initializing.")
    kwargs.setdefault('LOCAL_RANK', '0')
    os.environ.update(kwargs)
    if not os.path.lexists('data'):
        os.symlink('/data/oss_bucket_0', 'data')
    if not os.path.lexists('pretrained'):
        os.symlink('/data/oss_bucket_0/ckpts', 'pretrained')
    if not os.path.lexists('work_dirs'):
        os.symlink('/data/oss_bucket_0/work_dirs', 'work_dirs')
    logger.debug(f"ODPS initialization done with {os.listdir('.')}.")


class Runner:

    def __init__(
        self,
        name: str,
        config_file: pathlib.Path,
        config_options: Optional[todd.base.Config],
    ) -> None:
        self.name = name
        self.config = todd.base.Config.load(config_file)

        if config_options is not None:
            for k, v in config_options.items():
                todd.base.setattr_recur(self.config, k, v)

        self.work_dir = pathlib.Path(f'work_dirs/{name}')
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if todd.base.get_rank() == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = self.work_dir / f'{timestamp}.log'
            todd.base.init_log_file(log_file)
        self.logger = todd.base.get_logger()
        self.logger.info(f"Version: {todd.base.git_commit_id()}")

    def build_train_fixtures(self) -> None:
        self.criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=self.config.lr, steps_per_epoch=len(self.train_loader), epochs=self.config.epoch, pct_start=0.2,
        )

    def build_train_dataloader(self, transform: tf.Compose) -> None:
        transforms = list(transform.transforms)
        transforms.insert(2, mldec.datasets.CutoutPIL(cutout_factor=0.5))
        transforms.insert(3, RandAugment())
        self.train_transform = tf.Compose(transforms)
        self.train_dataset = mldec.datasets.CocoClassification(transform=self.train_transform, **self.config.train.dataset)
        self.train_sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        )
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.config.train.batch_size, sampler=self.train_sampler,
            num_workers=self.config.train.workers,
            )

    def build_val_dataloader(self, transform: tf.Compose) -> None:
        transforms = list(transform.transforms)
        self.val_transform = tf.Compose(transforms)
        self.val_dataset = mldec.datasets.PatchedCocoClassification(transform=self.val_transform, **self.config.val.dataset)
        self.val_sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.config.val.batch_size, sampler=self.val_sampler,
            num_workers=self.config.val.workers, collate_fn=self.val_dataset.collate,
            )

    def build_model(
        self,
        clip_model: clip.model.CLIP,
    ) -> None:
        model = mldec.coop.CustomCLIP(
            clip_model=clip_model,
            classnames=self.val_dataset.classnames,
            frozen_config=todd.base.Config(
                no_grad_config=dict(
                    names=[
                        '._text_encoder._clip_text_encoder',
                        '._image_encoder._clip_image_encoder',
                    ],
                ),
                eval_config=dict(
                    names=[
                        '._text_encoder._clip_text_encoder',
                        '._image_encoder._clip_image_encoder',
                    ],
                ),
            ),
        )
        model.float()
        model.requires_grad_()
        if not debug.CPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        self.model = model

    def save_checkpoint(self, epoch: int) -> None:
        todd.base.save_checkpoint(
            self.model, self.work_dir / f'epoch_{epoch}.pth',
            optimizer=self.optimizer, scheduler=self.scheduler,
        )

    def load_checkpoint(self, epoch: int) -> None:
        todd.base.load_checkpoint(
            self.model, self.work_dir / f'epoch_{epoch}.pth',
            optimizer=getattr(self, 'optimizer', None),
            scheduler=getattr(self, 'scheduler', None),
        )

    def train(self) -> float:
        record = -1
        for epoch in range(self.config.epoch):
            if not debug.CPU:
                torch.distributed.barrier()
                self.train_sampler.set_epoch(epoch)
            self.model.train()
            self.train_epoch(epoch)
            self.model.eval()
            mAP = self.val()
            if mAP > record:
                record = mAP
            if todd.base.get_rank() == 0:
                self.logger.info(f"record = {record * 100:.2f}")
        return record

    def train_epoch(self, epoch: int) -> None:
        for i, batch in enumerate(self.train_loader):
            loss = self.train_iter(*batch)
            if i % self.config.log_interval != 0:
                continue
            self.logger.info(
                f'Epoch [{epoch}/{self.config.epoch}] '
                f'Train Step [{i}/{len(self.train_loader)}] '
                f'LR {self.scheduler.get_last_lr()[0]:.2e} '
                f'Loss {loss:.1f}'
            )
            if debug.CPU:
                self.logger.debug(
                    f'Scaler {self.model._scaler.tolist()} '
                    f'Bias {self.model._bias.tolist()}'
                )
            else:
                self.logger.debug(
                    f'Scaler {self.model.module._scaler.tolist()} '
                    f'Bias {self.model.module._bias.tolist()}'
                )
            if debug.LESS_DATA and i: break

        if todd.base.get_rank() == 0:
            self.save_checkpoint(epoch)

    def train_iter(self, inputData: torch.Tensor, target: torch.Tensor) -> float:
        if not debug.CPU:
            inputData = inputData.cuda()
            target = target.cuda()
        outputs = self.model(inputData)
        loss = sum(self.criterion(output.sigmoid(), target) for output in outputs)
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    @torch.no_grad()
    def val(self) -> float:
        preds = []
        targets = []
        for i, batch in enumerate(self.val_loader):
            pred, target = self.val_iter(*batch)
            preds.append(pred)
            targets.append(target)
            if i % self.config.log_interval != 0:
                continue
            self.logger.info(
                f'Val Step [{i}/{len(self.val_loader)}]'
            )
            if debug.LESS_DATA and i: break
        if not debug.CPU:
            preds_ = torch.cat(preds)
            targets_ = torch.cat(targets)
            preds = [torch.zeros_like(preds_) for _ in range(todd.base.get_world_size())]
            targets = [torch.zeros_like(targets_) for _ in range(todd.base.get_world_size())]
            torch.distributed.all_gather(preds, preds_)
            torch.distributed.all_gather(targets, targets_)

        if todd.base.get_rank() == 0:
            preds_ = torch.cat(preds).cpu().numpy()
            targets_ = torch.cat(targets).cpu().numpy()
            mAP = sklearn.metrics.average_precision_score(targets_, preds_)
            self.logger.info(f"mAP = {mAP * 100:.2f}")
            return mAP
        else:
            return -1

    def val_iter(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        num_patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not debug.CPU:
            input = input.cuda()
            target = target.cuda()
        pred = self.model(input)
        pred_ = functools.reduce(torch.max, pred)
        pred_ = torch.stack(tuple(p.max(0).values for p in pred_.split(num_patches.tolist())))
        return pred_, target

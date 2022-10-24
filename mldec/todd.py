__all__ = [
    'BaseRunner',
    'TrainerMixin',
]

from abc import ABC, abstractmethod
import logging
import pathlib
import time
from typing import Any, Dict, List, Optional
import einops.layers.torch
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.distributed as dist
import torch.nn as nn


import todd

from .debug import debug

# fix logging format
import lvis
logger = logging.getLogger()
for handler in logger.handlers:
    logger.removeHandler(handler)


class BaseRunner(ABC):

    def __init__(
        self,
        *args,
        name: str,
        config: todd.Config,
        load=None,
        **kwargs,
    ) -> None:
        self._name = name
        self._config = config

        self._work_dir = self._build_work_dir(*args, config=config, **kwargs)
        self._logger = self._build_logger(
            *args,
            config=config.get('logger'),
            **kwargs,
        )
        self._dataset, self._sampler, self._dataloader = \
            self._build_dataloader(
                *args,
                config=todd.getattr_recur(
                    config,
                    '.val.dataloader',
                    None,
                ),
                **kwargs,
            )
        self._model = self._build_model(
            *args,
            config=config.get('model'),
            **kwargs,
        )

        self._build_custom(*args, config=config, **kwargs)

        self._epoch = -1
        if load is None:
            todd.init_iter()
        else:
            self.load_checkpoint(
                *args,
                epoch=load,
                **kwargs,
            )

    def _build_work_dir(
        self,
        *args,
        config: todd.Config,
        **kwargs,
    ) -> pathlib.Path:
        work_dir = pathlib.Path(f'work_dirs/{self._name}')
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def _build_logger(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> logging.Logger:
        init_log_file = config is None or config.get('init_log_file', True)
        if init_log_file and todd.get_rank() == 0:
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            log_file = self._work_dir / f'{timestamp}.log'
            todd.init_log_file(log_file)
        logger = todd.get_logger()
        logger.info(f"Version {todd.git_commit_id()}")
        logger.info(f"Config\n{self._config.dumps()}")
        return logger

    @abstractmethod
    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ):
        pass

    @abstractmethod
    def _build_model(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> nn.Module:
        pass

    def _build_custom(
        self,
        *args,
        config: todd.Config,
        **kwargs,
    ) -> None:
        pass

    def load_checkpoint(self, *args, epoch, config: Optional[todd.Config] = None, **kwargs) -> None:
        if config is None:
            config = todd.getattr_recur(self._config, '.checkpoint.load_', dict())
        todd.load_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
            **config,
        )
        todd.init_iter((epoch + 1) * len(self._dataloader))
        self._epoch = epoch

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        self._model.eval()
        return dict()

    def _before_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        pass

    @abstractmethod
    def _run_iter(self, *args, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        pass

    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        pass

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs):
        pass

    @torch.no_grad()
    def run(self, *args, **kwargs):
        memo = self._before_run(*args, **kwargs)
        for i, batch in enumerate(self._dataloader, 1):
            self._before_run_iter(*args, i=i, batch=batch, memo=memo, **kwargs)
            try:
                self._run_iter(*args, i=i, batch=batch, memo=memo, **kwargs)
            except Exception as e:
                self._logger.exception(
                    f"Unable to run iter {i}\n"
                    f"args={args}\n"
                    f"batch={batch}\n"
                    f"memo={memo}\n"
                    f"kwargs={kwargs}"
                )
                raise
            end = self._after_run_iter(
                *args,
                i=i,
                batch=batch,
                memo=memo,
                log=(i % todd.getattr_recur(self._config, '.logger.interval', 1) == 0),
                **kwargs,
            )
            if end:
                break
        return self._after_run(*args, memo=memo, **kwargs)


class TrainerMixin(BaseRunner):

    def _build_custom(self, *args, config: todd.Config, **kwargs) -> None:
        super()._build_custom(*args, config=config, **kwargs)
        self._train_dataset, self._train_sampler, self._train_dataloader = \
            self._build_train_dataloader(
                *args,
                config=todd.getattr_recur(
                    config,
                    '.train.dataloader',
                    None,
                ),
                **kwargs,
            )
        self._build_train_fixtures(
            *args,
            config=config.get('train'),
            **kwargs,
        )
        self._optimizer = self._build_optimizer(
            *args,
            config=todd.getattr_recur(
                config,
                '.train.optimizer',
                None,
            ),
            **kwargs,
        )
        self._scheduler = self._build_scheduler(
            *args,
            config=todd.getattr_recur(
                config,
                '.train.lr_scheduler',
                None,
            ),
            **kwargs,
        )

    @abstractmethod
    def _build_train_dataloader(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ):
        pass

    @abstractmethod
    def _build_train_fixtures(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> None:
        pass

    def _build_optimizer(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> Optional[optim.Optimizer]:
        if config is None:
            return None
        return todd.utils.OPTIMIZERS.build(
            config,
            default_args=dict(model=self._model),
        )

    def _build_scheduler(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        if config is None:
            return None
        return todd.utils.LR_SCHEDULERS.build(
            config,
            default_args=dict(optimizer=self._optimizer),
        )

    def load_checkpoint(self, *args, epoch, config: Optional[todd.Config] = None, **kwargs) -> None:
        if config is None:
            config = todd.getattr_recur(self._config, '.checkpoint.load_', dict())
        todd.load_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            **config,
        )
        todd.init_iter((epoch + 1) * len(self._train_dataloader))
        self._epoch = epoch

    def save_checkpoint(self, *args, epoch, config: Optional[todd.Config] = None, **kwargs) -> None:
        if config is None:
            config = todd.getattr_recur(self._config, '.checkpoint.save', dict())
        todd.save_checkpoint(
            self._model, self._work_dir / f'epoch_{epoch}.pth',
            optimizer=self._optimizer, scheduler=self._scheduler,
            **config,
        )

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        return dict()

    def _before_train_epoch(self, *args, epoch: int, memo: Dict[str, Any], **kwargs) -> None:
        if isinstance(self._train_sampler, data.DistributedSampler):
            self._train_sampler.set_epoch(epoch)
        self._model.train()

    def _before_train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        pass

    @abstractmethod
    def _train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], **kwargs) -> None:
        pass

    def _after_train_iter(self, *args, epoch: int, i: int, batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        pass

    def _after_train_epoch(self, *args, epoch: int, memo: Dict[str, Any], **kwargs) -> Optional[bool]:
        pass

    def _after_train(self, *args, memo: Dict[str, Any], **kwargs):
        pass

    def train(self, *args, **kwargs):
        memo = self._before_train(*args, **kwargs)
        for epoch in range(self._epoch + 1, self._config.train.epoch):
            if not debug.CPU:
                dist.barrier()
            self._before_train_epoch(*args, epoch=epoch, memo=memo, **kwargs)
            for i, batch in enumerate(self._train_dataloader, 1):
                self._before_train_iter(*args, epoch=epoch, i=i, batch=batch, memo=memo, **kwargs)
                try:
                    self._train_iter(*args, epoch=epoch, i=i, batch=batch, memo=memo, **kwargs)
                except Exception as e:
                    self._logger.exception(
                        f"Unable to train iter {i}\n"
                        f"args={args}\n"
                        f"batch={batch}\n"
                        f"memo={memo}\n"
                        f"kwargs={kwargs}\n"
                    )
                    raise
                end = self._after_train_iter(
                    *args,
                    epoch=epoch,
                    i=i,
                    batch=batch,
                    memo=memo,
                    log=(i % todd.getattr_recur(self._config, '.logger.interval', 1) == 0),
                    **kwargs,
                )
                if end:
                    break
            end = self._after_train_epoch(*args, epoch=epoch, memo=memo, **kwargs)
            if end:
                break
        return self._after_train(*args, memo=memo, **kwargs)


@todd.losses.LOSSES.register_module()
class HierGKDLoss(todd.losses.functional.CrossEntropyLoss):

    def __init__(self, *args, temperature=10.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._embed = nn.Sequential(
            nn.AdaptiveMaxPool2d((3, 3)),
            einops.layers.torch.Rearrange('b c h w -> b c (h w)', h=3, w=3),
        )
        self._temperature = temperature

    def forward(
        self,
        preds: List[torch.Tensor],
        targets: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute HierKD GKD loss.

        Refer to http://arxiv.org/abs/2203.10593.

        Args:
            preds: [b x c x h x w]
            targets: b x c

        Returns:
            loss: 1
        """
        preds = list(map(self._embed, preds))
        fused_preds = F.normalize(torch.cat(preds, -1))
        attn_weights: torch.Tensor = torch.einsum('a c, b c n -> a b n', targets, fused_preds)
        attn_weights = attn_weights.softmax(-1)
        values = torch.einsum('a b n, b c n -> a c b', attn_weights, fused_preds)
        values = F.normalize(values)
        logits: torch.Tensor = torch.einsum('a c, a c b -> a b', targets, values)
        assert logits.shape[0] == logits.shape[1]
        logits = self._temperature * logits
        cl_targets = torch.arange(logits.shape[0], device=logits.device)
        return (
            super().forward(logits, cl_targets, *args, **kwargs)
            + super().forward(logits.T, cl_targets, *args, **kwargs)
        )


@todd.losses.LOSSES.register_module()
class RKDLoss(todd.losses.functional.MSELoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_relations(self, feats: torch.Tensor) -> torch.Tensor:
        """Get relations between each pair of feats.

        Args:
            feats: * x c

        Returns:
            relations: prod(*) x prod(*)
        """
        feats = feats.reshape(-1, feats.shape[-1])
        feats = F.normalize(feats)
        relations = torch.einsum('m c, n c -> m n', feats, feats)
        return relations

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Compute RKD loss.

        Args:
            preds: * x c
            targets: * x d

        Returns:
            loss: 1
        """
        assert preds.shape[:-1] == targets.shape[:-1]
        pred_relations = self.get_relations(preds)
        target_relations = self.get_relations(targets)
        return super().forward(
            pred_relations,
            target_relations,
            *args,
            **kwargs,
        )

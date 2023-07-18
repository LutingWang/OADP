__all__ = [
    'Control',
    'BaseRunner',
    'Validator',
    'Trainer',
    'IterBasedTrainer',
    'EpochBasedTrainer',
    'RunnerRegistry',
    'Memo',
]

import enum
import getpass
import itertools
import logging
import pathlib
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable, Dict

import torch
import torch.distributed
import torch.utils.data

from ..base import (
    Config,
    Formatter,
    LrSchedulerRegistry,
    OptimizerRegistry,
    Registry,
    Store,
)

Memo = Dict[str, Any]


@runtime_checkable
class DistributedSamplerProto(Protocol):

    def set_epoch(self, epoch: int) -> None:
        ...


class Control(enum.Enum):
    BREAK = enum.auto()
    CONTINUE = enum.auto()


class BaseRunner(ABC):

    def __init__(
        self,
        name: str,
        model: torch.nn.Module,
        dataloader: Config,
        log: Config,
        load_state_dict: Config,
        state_dict: Config,
    ) -> None:
        self._work_dir = self._build_work_dir(name)
        self._model = model
        self._logger = self._build_logger()

        self._dataloader = self._build_dataloader(dataloader)
        self._log = log
        self._load_state_dict = load_state_dict
        self._state_dict = state_dict

        self._iter = 0
        self._logger.debug(
            f"Runner initialized by {getpass.getuser()}@{socket.gethostname()}"
        )

    @property
    def iters(self) -> int:
        return len(self._dataloader)

    def _build_logger(self) -> logging.Logger:
        file = self._work_dir / '_.log'
        file = file.with_stem(datetime.now().astimezone().isoformat()) # type: ignore
        handler = logging.FileHandler(file)
        handler.setFormatter(Formatter())

        from ..base import logger
        name = f'{logger.name}.{self.__class__.__name__}'
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        return logger

    def _build_work_dir(self, name: str) -> pathlib.Path:
        work_dir = pathlib.Path('work_dirs') / name
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    @property
    def model(self) -> torch.nn.Module:
        """The underlying model if being wrapped."""
        model = self._model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        return model

    @abstractmethod
    def _build_dataloader(self, config: Config) -> torch.utils.data.DataLoader:
        """Build the dataloader.

        Args:
            config: dataloader config.
        """
        pass

    def _control_run_iter(self, batch, memo: Memo):
        """Whether the current iteration should execute or not.

        Args:
            i: iteration index.
            batch: inputs.
            memo: runtime memory.

        Returns:
            The control action or `None` for no control.

        Override this method for early stopping, error detection, etc.
        By default, this method supports `DRY_RUN` by returning
        `Control.BREAK` after the first log message.
        """
        if Store.DRY_RUN and self._iter > self._log.interval:
            return Control.BREAK
        return None

    def _before_run_iter(self, batch, memo: Memo) -> None:
        pass

    def _before_run_iter_log(self, batch, memo: Memo):
        pass

    @abstractmethod
    def _run_iter(self, batch, memo: Memo) -> torch.Tensor:
        pass

    def _after_run_iter_log(self, batch, memo: Memo):
        loss: torch.Tensor = memo['loss']
        return f"Iter [{self._iter}/{self.iters}] Loss {loss.item():.3f}"

    def _after_run_iter(self, batch, memo: Memo) -> None:
        pass

    def _before_run(self) -> Memo:
        return dict(dataloader=self._dataloader)

    def _run(self, memo: Memo) -> None:
        dataloader = memo['dataloader']
        for batch in dataloader:
            self._iter += 1
            control = self._control_run_iter(batch, memo)
            if control is Control.BREAK:
                break
            if control is Control.CONTINUE:
                continue
            assert control is None, control
            self._before_run_iter(batch, memo)
            if log := self._iter % self._log.interval == 0:
                info = self._before_run_iter_log(batch, memo)
                if info is not None:
                    self._logger.info(info)
            try:
                memo['loss'] = self._run_iter(batch, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run iter {self._iter}\n{batch=}\n{memo=}"
                )
                raise
            if log:
                info = self._after_run_iter_log(batch, memo)
                if info is not None:
                    self._logger.info(info)
            self._after_run_iter(batch, memo)

    def _after_run(self, memo: Memo) -> None:
        pass

    def run(self) -> Memo:
        memo = self._before_run()
        self._run(memo)
        self._after_run(memo)
        return memo

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.model.load_state_dict(
            state_dict['model'],
            **self._load_state_dict.model,
        )
        self._iter = state_dict['iter_']

    def state_dict(self):
        model = self.model.state_dict(**self._state_dict.model)
        return dict(model=model, iter_=self._iter)

    def write_state_dict(self, f: pathlib.Path) -> None:
        self._logger.info(f"Writing state dict to {f}")
        torch.save(self.state_dict(), f)

    def read_state_dict(self, f: pathlib.Path) -> None:
        self._logger.info(f"Reading state dict from {f}")
        self.load_state_dict(torch.load(f, 'cpu'))


class Validator(BaseRunner):

    def _before_run(self) -> Memo:
        self._model.eval()
        return super()._before_run()

    @torch.no_grad()
    def run(self) -> Memo:
        return super().run()


class Trainer(BaseRunner):

    def __init__(
        self,
        *args,
        optimizer: Config,
        lr_scheduler = None,
        validator = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._optimizer = self._build_optimizer(optimizer)
        self._lr_scheduler = (
            None if lr_scheduler is None else
            self._build_lr_scheduler(lr_scheduler)
        )
        self._validator = (
            None if validator is None else RunnerRegistry.build(validator)
        )

    def _build_optimizer(self, config: Config) -> torch.optim.Optimizer:
        return OptimizerRegistry.build(config, Config(model=self.model))

    def _build_lr_scheduler(
        self,
        config: Config,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return LrSchedulerRegistry.build(config)

    def _after_run_iter(self, batch, memo: Memo) -> None:
        super()._after_run_iter(batch, memo)
        loss: torch.Tensor = memo['loss']
        loss.backward()
        self._optimizer.step()
        self._model.zero_grad()

    def _after_run_iter_log(self, batch, memo: Memo):
        info = super()._after_run_iter_log(batch, memo)
        if self._lr_scheduler is None:
            return info
        last_lr = [f'{lr:.3e}' for lr in self._lr_scheduler.get_last_lr()]
        info = "" if info is None else f"{info} "
        info += f"LR {last_lr}"
        return info

    def _before_run(self) -> Memo:
        self._model.train()
        return super()._before_run()

    def _after_run(self, memo: Memo) -> None:
        super()._after_run(memo)
        self.validate()

    def validate(self) -> None:
        if self._validator is None:
            self._logger.info(
                "Skipping validation since validator is undefined"
            )
            return
        torch.distributed.barrier()
        self._validator.run()
        self._model.train()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._optimizer.load_state_dict(
            state_dict['optimizer'],
            **self._load_state_dict.optimizer,
        )
        if self._lr_scheduler is not None:
            self._lr_scheduler.load_state_dict(
                state_dict['lr_scheduler'],
                **self._load_state_dict.lr_scheduler,
            )

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['optimizer'] = self._optimizer.state_dict(
            **self._state_dict.optimizer,
        )
        if self._lr_scheduler is not None:
            state_dict['lr_scheduler'] = self._lr_scheduler.state_dict(
                **self._state_dict.lr_scheduler,
            )
        return state_dict


class IterBasedTrainer(Trainer):

    def __init__(self, *args, iters: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iters = iters

    @property
    def iters(self) -> int:
        return self._iters

    def _after_run_iter(self, batch, memo: Memo) -> None:
        super()._after_run_iter(batch, memo)
        if self._iter % self._state_dict.interval == 0:
            self.validate()
            self.write_state_dict(self._work_dir / f'iter_{self._iter}.pth')
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _before_run(self) -> Memo:
        memo = super()._before_run()
        memo['dataloader'] = itertools.islice(
            itertools.cycle(memo['dataloader']),
            self._iters,
        )
        return memo

    def _after_run(self, memo: Memo) -> None:
        self.write_state_dict(self._work_dir / 'latest.pth')
        super()._after_run(memo)


class EpochBasedTrainer(Trainer):

    def __init__(self, *args, epochs: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._epoch = 0
        self._epochs = epochs

    def _control_run_epoch(self, memo: Memo):
        return None

    def _before_run_epoch(self, memo: Memo) -> Memo:
        sampler = self._dataloader.batch_sampler
        if sampler is None:
            sampler = self._dataloader.sampler
        if isinstance(sampler, DistributedSamplerProto):
            sampler.set_epoch(self._epoch)
        return super()._before_run()

    def _before_run_epoch_log(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ):
        return f"Epoch [{self._epoch}/{self._epochs}] beginning"

    def _run_epoch(self, epoch_memo: Memo, memo: Memo) -> None:
        super()._run(epoch_memo)

    def _after_run_epoch_log(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ):
        return f"Epoch [{self._epoch}/{self._epochs}] ended"

    def _after_run_epoch(
        self,
        epoch_memo: Memo,
        memo: Memo,
    ) -> None:
        super()._after_run(epoch_memo)
        memo['epoch_memos'][self._epoch] = epoch_memo
        self.write_state_dict(self._work_dir / f'epoch_{self._epoch}.pth')
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _before_run(self) -> Memo:
        memo: Memo = dict(epoch_memos=dict())
        return memo

    def _run(self, memo: Memo) -> None:
        while self._epoch < self._epochs:
            self._epoch += 1
            self._iter = 0

            control = self._control_run_epoch(memo)
            if control is Control.BREAK:
                break
            if control is Control.CONTINUE:
                continue
            assert control is None, control
            epoch_memo = self._before_run_epoch(memo)
            info = self._before_run_epoch_log(epoch_memo, memo)
            if info is not None:
                self._logger.info(info)
            try:
                self._run_epoch(epoch_memo, memo)
            except Exception:
                self._logger.exception(
                    f"Unable to run epoch {self._epoch}\n{epoch_memo=}\n"
                    f"{memo=}"
                )
                raise
            info = self._after_run_epoch_log(epoch_memo, memo)
            if info is not None:
                self._logger.info(info)
            self._after_run_epoch(epoch_memo, memo)

    def _after_run(self, memo: Memo) -> None:
        pass


class RunnerRegistry(Registry):
    pass

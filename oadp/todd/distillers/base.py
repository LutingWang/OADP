__all__ = [
    'DistillerRegistry',
    'BaseDistiller',
    'DistillerStore',
]

import warnings
from typing import Callable, Iterable, Mapping, cast
from typing_extensions import Self

import torch.nn as nn

from ..adapts import AdaptRegistry
from ..base import (
    Config,
    Job,
    Message,
    Module,
    ModuleList,
    Registry,
    Spec,
    StoreMeta,
    Workflow,
    transfer_weights,
)
from ..hooks import BaseHook, HookRegistry
from ..losses import LossRegistry
from ..losses.base import BaseLoss


class DistillerStore(metaclass=StoreMeta):
    CHECK_INPUTS: bool
    INTERMEDIATE_OUTPUTS: str


class BaseDistiller(Module, Workflow):

    @classmethod
    def build(cls, config: Config) -> Self:
        config = config.copy()

        models = tuple(config.pop('models'))

        hooks: Config = config.pop('hooks')
        adapts: Config = config.pop('adapts')
        losses: Config = config.pop('losses')

        weight_transfer = config.pop('weight_transfer', None)

        hook_job = Job([])
        for k, v in hooks.items():
            hook = Job.build(Config(registry=HookRegistry, steps=v))
            for action in hook.actions:
                cast(BaseHook, action).bind(models[k])
            hook_job += hook

        adapt_job = Job.build(Config(registry=AdaptRegistry, steps=adapts))
        loss_job = Job.build(Config(registry=LossRegistry, steps=losses))

        jobs = dict(hooks=hook_job, adapts=adapt_job, losses=loss_job)
        distiller = cls(models, jobs, **config)

        distiller.add_module(
            '_adapts',
            ModuleList(ModuleList(step.actions) for step in adapt_job),
        )
        distiller.add_module(
            '_losses',
            ModuleList(ModuleList(step.actions) for step in loss_job),
        )

        if weight_transfer is not None:
            transfer_weights(distiller, weight_transfer)

        return distiller

    def __init__(
        self,
        models: Iterable[nn.Module],
        jobs: Mapping[str, Job],
    ) -> None:
        Module.__init__(self)
        Workflow.__init__(self, jobs)  # type: ignore[arg-type]
        self._models = tuple(models)

        outputs: set[str] = set()
        for hook in self['hooks']:
            spec = hook.spec
            assert len(spec.inputs) == 0
            assert outputs.isdisjoint(spec.outputs)
            outputs |= spec.outputs

    def __hash__(self) -> int:
        # self inherits from UserDict, which defines `__eq__`, so `__hash__`
        # is disabled by default
        return id(self)

    def __call__(self, message = None) -> Message:
        if message is None:
            message = dict()

        if DistillerStore.CHECK_INPUTS:
            spec = self.spec
            inputs = message.keys()
            if len(spec.inputs ^ inputs):
                warnings.warn(
                    f"Missing inputs {spec.inputs - inputs}\n"
                    f"Unexpected inputs {inputs - spec.inputs}\n"
                )

        tensors = self.tensors()
        if message is not None:
            tensors.update(message)
        self['adapts'](tensors)
        losses = self['losses'](tensors.copy())

        if DistillerStore.INTERMEDIATE_OUTPUTS:
            losses[DistillerStore.INTERMEDIATE_OUTPUTS] = tensors

        return losses

    @property
    def spec(self) -> Spec:
        hook_spec = self['hooks'].spec
        adapt_spec = self['adapts'].spec
        loss_spec = self['losses'].spec
        return Spec(
            (loss_spec.inputs - adapt_spec.outputs)
            | adapt_spec.inputs - hook_spec.outputs,
            loss_spec.outputs,
        )

    @property
    def models(self):
        return self._models

    def _apply(self, fn: Callable[..., None]) -> Self:
        for model in self._models:
            if getattr(model, 'sync_apply', True):
                model._apply(fn)
        return super()._apply(fn)

    def track_tensors(self) -> None:
        hooks = self['hooks'].actions
        for hook in filter(lambda hook: hook.tracking_mode, hooks):
            hook.track_tensor()

    def tensors(self) -> Message:
        tensors: Message = dict()
        self['hooks'](tensors)
        return tensors

    def reset(self) -> None:
        hooks = self['hooks'].actions
        for hook in hooks:
            hook.reset()

    def step(self) -> None:
        for loss in self['losses'].actions:
            cast(BaseLoss, loss)._weight.step()


class DistillerRegistry(Registry):

    @classmethod
    def _build(cls, config: Config) -> BaseDistiller:
        distiller: type[BaseDistiller] = cls[config.pop('type')]
        return distiller.build(config)

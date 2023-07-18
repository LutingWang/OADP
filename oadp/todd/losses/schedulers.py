__all__ = [
    'BaseScheduler',
    'SchedulerRegistry',
    'ConstantScheduler',
    'WarmupScheduler',
    'EarlyStopScheduler',
    'DecayScheduler',
    'StepScheduler',
    'CosineAnnealingScheduler',
    'ChainedScheduler',
]

import bisect
import math
from abc import ABC, abstractmethod
from typing import Iterable

import torch

from ..base import Registry


class BaseScheduler(torch.nn.Module, ABC):
    """Base class for schedulers.

    Under most cases, schedulers are used as a variable loss weight.
    Schedulers are functions of `steps`, which could mean iterations or
    epochs.
    Users could increment `steps` by calling `step`, or directly set the
    `steps` property.
    Call the scheduler to get the value for the current step.

    .. note:: `steps` starts from 1, so `step` should be called after the
        first step.
    """

    def __init__(self, gain: float = 1.0) -> None:
        """Initialize.

        Args:
            gain: multiplier to the scheduler value.
        """
        super().__init__()
        self._gain = gain
        self.register_forward_hook(forward_hook) # type: ignore
        self.register_buffer('_steps', torch.tensor(1))

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def steps(self) -> int:
        return self._steps.item() # type: ignore

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = torch.tensor(value)

    def step(self) -> None:
        self._steps += 1

    @abstractmethod
    def forward(self) -> float:
        """The scheduler's function.

        Returns:
            The scheduler's value for the current step, before multiplying
            `gain`.

        Since `gain` is handled by this base class, it is usually adequate for
        `forward` to return a percentage value in :math:`[0, 1]`.
        """
        pass


def forward_hook(module: BaseScheduler, input_, output: float) -> float:
    return output * module.gain


class SchedulerRegistry(Registry):
    pass


@SchedulerRegistry.register()
class ConstantScheduler(BaseScheduler):
    """Unvarying scheduler.

    The value of this scheduler is always the `gain`:

        >>> constant = ConstantScheduler(5)
        >>> constant()
        5.0
        >>> constant.step()
        >>> constant()
        5.0
    """

    def forward(self) -> float:
        return 1.0


@SchedulerRegistry.register()
class WarmupScheduler(BaseScheduler):
    """Warmup scheduler.

    The value will linearly increase from 0 to 1.
    At step ``end``, the value is 1.

        >>> warmup = WarmupScheduler(end=5)
        >>> for _ in range(7):
        ...     print(warmup())
        ...     warmup.step()
        0.2
        0.4
        0.6
        0.8
        1.0
        1.0
        1.0
    """

    def __init__(self, *args, end: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._end = end

    def forward(self) -> float:
        return min(self.steps / self._end, 1.0)


@SchedulerRegistry.register()
class EarlyStopScheduler(BaseScheduler):
    """Early stop.

    At some point, the value drops to 0 from 1.

        >>> early_stop = EarlyStopScheduler(at=3)
        >>> for _ in range(5):
        ...     print(early_stop())
        ...     early_stop.step()
        1.0
        1.0
        0.0
        0.0
        0.0
    """

    def __init__(self, *args, at: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._at = at

    def forward(self) -> float:
        return float(self.steps < self._at)


@SchedulerRegistry.register()
class DecayScheduler(BaseScheduler):
    """Decay scheduler.

    Before or at ``start``, the value is 1.
    After or at ``end``, the value is 0.
    In between, the value is interpolated.

        >>> decay = DecayScheduler(start=2, end=7)
        >>> for _ in range(8):
        ...     print(decay())
        ...     decay.step()
        1.0
        1.0
        0.8
        0.6
        0.4
        0.2
        0.0
        0.0
    """

    def __init__(self, *args, start: int = 1, end: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._start = start
        self._end = end

    def forward(self) -> float:
        if self.steps <= self._start:
            return 1.0
        if self.steps >= self._end:
            return 0.0
        return (self._end - self.steps) / (self._end - self._start)


@SchedulerRegistry.register()
class StepScheduler(BaseScheduler):
    """Step scheduler.

    The value is multiplied by :math:`gamma` at every milestone:

        >>> step = StepScheduler(milestones=[3, 4], gamma=0.1)
        >>> for _ in range(5):
        ...     print(round(step(), 2))
        ...     step.step()
        1.0
        1.0
        0.1
        0.01
        0.01
    """

    def __init__(
        self,
        *args,
        milestones: Iterable[int],
        gamma: float,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._milestones = sorted(milestones)
        self._gamma = gamma

    def forward(self) -> float:
        return self._gamma**bisect.bisect(self._milestones, self.steps)


@SchedulerRegistry.register()
class CosineAnnealingScheduler(BaseScheduler):
    """Cosine annealing scheduler.

    The value anneals as the cosine function is defined.
    The first step starts with 1.
    After ``duration`` steps, the value becomes 0.
    The best practice is to set ``duration`` to the total number of steps.

        >>> cosine = CosineAnnealingScheduler(duration=5)
        >>> for _ in range(6):
        ...     print(round(cosine(), 6))
        ...     cosine.step()
        1.0
        0.904508
        0.654508
        0.345492
        0.095492
        0.0
    """

    def __init__(self, *args, duration: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._duration = duration

    def forward(self) -> float:
        steps = self.steps - 1
        if steps >= self._duration:
            return 0
        return (1 + math.cos(math.pi * steps / self._duration)) / 2


@SchedulerRegistry.register()
class ChainedScheduler(BaseScheduler):
    """Chained scheduler.

    Schedulers are chained in an multiplicative manner:

        >>> warmup = WarmupScheduler(end=5, gain=10)
        >>> step = StepScheduler(milestones=[3, 4], gamma=0.1)
        >>> chained = ChainedScheduler(schedulers=[warmup, step])
        >>> for _ in range(5):
        ...     print(round(chained(), 6))
        ...     chained.step()
        2.0
        4.0
        0.6
        0.08
        0.1
    """

    def __init__(
        self,
        *args,
        schedulers: Iterable[BaseScheduler],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._schedulers = tuple(schedulers)

    def forward(self) -> float:
        return math.prod(scheduler() for scheduler in self._schedulers)

    def step(self) -> None:
        super().step()
        for scheduler in self._schedulers:
            scheduler.step()

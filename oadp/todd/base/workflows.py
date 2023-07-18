__all__ = [
    'Message',
    'Spec',
    'Task',
    'Step',
    'SingleStep',
    'ParallelStep',
    'ParallelSingleStep',
    'Job',
    'Workflow',
]

from abc import ABC, abstractmethod
from collections import UserDict, UserList
from functools import partial
from itertools import repeat
from symtable import symtable
from typing import TYPE_CHECKING, Any, Callable, Iterable, NamedTuple, final, Dict, Set, Tuple
from typing_extensions import Self

import pandas as pd

from .configs import Config
from .patches import logger
from .types import RegistryMeta

Message = Dict[str, Any]


class Spec(NamedTuple):
    inputs: Set[str]
    outputs: Set[str]


class Task(ABC):
    """Base class for `Step`, `Job`, and `Workflow`."""

    @classmethod
    @abstractmethod
    def build(cls, config: Config) -> Self:
        """Builds a task.

        Args:
            config: task configuration.

        Returns:
            The built task.
        """
        pass

    @abstractmethod
    def __call__(self, message: Message) -> Message:
        """Executes the task.

        Args:
            message: inputs.

        Returns:
            Outputs.
        """
        pass

    @property
    @abstractmethod
    def actions(self) -> Tuple[Callable, ...]:
        """User-defined actions used by the task."""
        pass

    @property
    @abstractmethod
    def spec(self) -> Spec:
        """Specifications of the task."""
        pass


class Step(Task):
    """Executor of actions."""

    @classmethod
    @final
    def build(cls, config: Config) -> 'Step':
        """Builds a step.

        Args:
            config: metadata of the step and specifications of the
                user-defined action.

        Returns:
            The built step.

        The user-defined action must be `typing.Callable`:

            >>> from todd import RegistryMeta
            >>> class Model(metaclass=RegistryMeta): pass
            >>> @Model.register()
            ... class ResNet:
            ...     def __init__(
            ...         self,
            ...         layers: int,
            ...         num_classes: int = 1024,
            ...     ) -> None:
            ...         self._layers = layers
            ...         self._num_classes = num_classes
            ...     def __call__(self, *args, **kwargs): ...
            ...     def __repr__(self) -> str:
            ...         return (
            ...             f"{type(self).__name__}("
            ...             f"layers={self._layers}, "
            ...             f"num_classes={self._num_classes})"
            ...         )

        To build a step of ``ResNet``, the ``config`` must specify the
        ``inputs`` and ``outputs`` of the step:

            >>> config = Config(
            ...     inputs=('x',),
            ...     outputs='y',
            ...     registry=Model,
            ...     action=dict(type='ResNet', layers=50),
            ... )
            >>> Step.build(config)
            SingleStep(inputs=('x',), outputs='y', action=ResNet(layers=50, \
num_classes=1024))

        By default, the built step is an instance of :py:class:`SingleStep`.
        When the ``config`` provides a ``parallel`` parameter, instances of
        :py:class:`ParallelStep` or :py:class:`ParallelSingleStep` will be
        built.

            >>> config.parallel = True
            >>> Step.build(config)
            ParallelSingleStep(inputs=('x',), outputs='y', action=ResNet(\
layers=50, num_classes=1024))
            >>> config.parallel = [dict(num_classes=10), dict(num_classes=100)]
            >>> Step.build(config)
            ParallelStep(inputs=('x',), outputs='y', actions=(ResNet(layers=50\
, num_classes=10), ResNet(layers=50, num_classes=100)))
        """
        config = config.copy()
        registry: RegistryMeta = config.pop('registry')
        action: Config = config.pop('action')
        build = partial(registry.build, action)

        parallel = config.pop('parallel', False)
        if isinstance(parallel, Iterable):
            return ParallelStep(tuple(map(build, parallel)), **config)
        if isinstance(parallel, int) and not isinstance(parallel, bool):
            return ParallelStep(
                tuple(build() for _ in range(parallel)),
                **config,
            )
        if parallel:
            return ParallelSingleStep(build(), **config)
        return SingleStep(build(), **config)

    def __init__(
        self,
        inputs: Iterable[str],
        outputs: str,
    ) -> None:
        """Initialize.

        Args:
            inputs: names of the input fields.
            outputs: expression of the outputs.

        For convenience, ``outputs`` is designed to be an expression.
        Suppose ``outputs`` is ``a, b``, the behavior of the step is similar
        to the following code:

        .. code-block:: python

           a, b = action(...)
           return dict(a=a, b=b)
        """
        self._inputs = tuple(inputs)
        self._outputs = outputs

    @property
    def name(self) -> str:
        """Name of the step.

        By default, this is the output of the step:

            >>> step = SingleStep(lambda: None, ['a', 'b'], 'c, d')
            >>> step.name
            'c, d'
        """
        return self._outputs

    @property
    def spec(self) -> Spec:
        """Specification of the step.

        A specification is a tuple of the input and output names:

            >>> step = SingleStep(lambda: None, ['a', 'b'], 'c, d')
            >>> spec = step.spec
            >>> sorted(spec.inputs)
            ['a', 'b']
            >>> sorted(spec.outputs)
            ['c', 'd']
        """
        return Spec(
            set(self._inputs),
            set(symtable(self._outputs, '<string>', 'eval').get_identifiers()),
        )

    def _input(self, message: Message) -> tuple:
        """Parse the inputs.

        Args:
            message: the original message.

        Returns:
            The parsed inputs.

        Convert ``self._inputs`` to the corresponding ``message`` fields:

            >>> step = Config(_inputs=('a', 'b'))
            >>> Step._input(step, dict(a=1, b=2, d=4))
            (1, 2)
        """
        return tuple(message[input_] for input_ in self._inputs)

    def _output(self, outputs) -> Message:
        """Parse the outputs.

        Args:
            outputs: outputs of the action.

        Returns:
            The parsed outputs.

        Parse according to `self._outputs`, which is arbitrary expression:

            >>> step = Config(_outputs='a, b')
            >>> Step._output(step, (1, 2))
            {'a': 1, 'b': 2}
        """
        message: Message = dict()
        exec(f'{self._outputs} = __o', dict(__o=outputs), message)
        return message


class SingleStep(Step):

    def __init__(self, action: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._action = action

    def __call__(self, message: Message) -> Message:
        inputs = self._input(message)
        _ = self._action(*inputs)
        outputs = self._output(_)
        return outputs

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"inputs={self._inputs}, "
            f"outputs={repr(self._outputs)}, "
            f"action={repr(self._action)})"
        )

    @property
    def actions(self) -> Tuple[Callable]:
        return self._action,


class ParallelStep(Step):

    def __init__(self, actions: Iterable[Callable], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._actions = actions

    def __call__(self, message: Message) -> Message:
        inputs = self._input(message)
        outputs = []
        for action, *_ in zip(self._actions, *inputs):
            _ = action(*_)
            outputs.append(self._output(_))
        return pd.DataFrame(outputs).to_dict(orient="list") # type: ignore

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"inputs={self._inputs}, "
            f"outputs={repr(self._outputs)}, "
            f"actions={repr(self._actions)})"
        )

    @property
    def actions(self) -> Tuple[Callable, ...]:
        return tuple(self._actions)


class ParallelSingleStep(ParallelStep):
    if TYPE_CHECKING:
        _actions: repeat[Callable]

    def __init__(self, action: Callable, *args, **kwargs) -> None:
        super().__init__(repeat(action), *args, **kwargs)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"inputs={self._inputs}, "
            f"outputs={repr(self._outputs)}, "
            f"action={repr(self.actions[0])})"
        )

    @property
    def actions(self) -> Tuple[Callable]:
        return next(self._actions),


class Job(UserList, Task):

    @classmethod
    def build(cls, config: Config) -> Self:
        config = config.copy()
        steps: Config = config.pop('steps')
        return cls([
            Step.build(Config(**v, **config, outputs=k))
            for k, v in steps.items()
        ])

    def __call__(self, message: Message) -> Message:
        updates: Message = dict()
        for step in self:
            try:
                outputs = step(message)
            except Exception:
                logger.error(f"Failed to forward {step}")
                raise
            message.update(outputs)
            updates.update(outputs)
        return updates

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({super().__repr__()})")

    @property
    def actions(self) -> Tuple[Callable, ...]:
        """Actions of all steps.

        A flat list of actions in sequence:

            >>> class Action:
            ...     def __init__(self, name: str) -> None:
            ...         self._name = name
            ...     def __call__(self) -> None:
            ...         pass
            ...     def __repr__(self) -> str:
            ...         return f"Action({repr(self._name)})"
            >>> job = Job([
            ...     SingleStep(Action('a'), [], ''),
            ...     ParallelStep([Action('b'), Action('c')], [], ''),
            ...     ParallelSingleStep(Action('d'), [], ''),
            ... ])
            >>> job.actions
            (Action('a'), Action('b'), Action('c'), Action('d'))
        """
        return sum((step.actions for step in self), tuple())

    @property
    def spec(self) -> Spec:
        inputs: set[str] = set()
        outputs: set[str] = set()
        for step in self:
            spec = step.spec
            inputs |= spec.inputs - outputs
            outputs |= spec.outputs
        return Spec(inputs, outputs)


class Workflow(UserDict, Task):

    @property
    def actions(self) -> Tuple[Callable, ...]:
        return sum((job.actions for job in self.values()), tuple())

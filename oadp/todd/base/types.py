__all__ = [
    'NonInstantiableMeta',
    'StoreMeta',
    'Store',
    'RegistryMeta',
    'Registry',
    'NormRegistry',
    'LrSchedulerRegistry',
    'OptimizerRegistry',
    'build_param_group',
    'build_param_groups',
]

import inspect
import os
import re
from collections import UserDict
from typing import Callable, Iterable, NoReturn, Sequence, TypeVar, Dict, List, Tuple, Optional, Union

import torch
import torch.distributed
import torch.nn as nn
from packaging.version import parse

from .configs import Config
from .patches import logger

T = TypeVar('T', bound=Callable)


class NonInstantiableMeta(type):

    def __call__(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError


class StoreMeta(NonInstantiableMeta):
    """Stores for global variables.

    Stores provide an interface to access global variables:

        >>> class CustomStore(metaclass=StoreMeta):
        ...     VARIABLE: int
        >>> CustomStore.VARIABLE
        0
        >>> CustomStore.VARIABLE = 1
        >>> CustomStore.VARIABLE
        1

    Variables cannot have the same name:

        >>> class AnotherStore(metaclass=StoreMeta):
        ...     VARIABLE: int
        Traceback (most recent call last):
        ...
        TypeError: Duplicated keys={'VARIABLE'}

    Variables can have explicit default values:

        >>> class DefaultStore(metaclass=StoreMeta):
        ...     DEFAULT: float = 0.625
        >>> DefaultStore.DEFAULT
        0.625

    Non-empty environment variables are read-only.
    For string variables, their values are read directly from the environment.
    Other environment variables are evaluated and should be of the
    corresponding type.
    Default values are ignored.

        >>> os.environ['ENV_INT'] = '2'
        >>> os.environ['ENV_STR'] = 'hello world!'
        >>> os.environ['ENV_DICT'] = 'dict(a=1)'
        >>> class EnvStore(metaclass=StoreMeta):
        ...     ENV_INT: int = 1
        ...     ENV_STR: str
        ...     ENV_DICT: dict
        >>> EnvStore.ENV_INT
        2
        >>> EnvStore.ENV_STR
        'hello world!'
        >>> EnvStore.ENV_DICT
        {'a': 1}

    Assignments to those variables will not trigger exceptions, but will not
    take effect:

        >>> EnvStore.ENV_INT = 3
        >>> EnvStore.ENV_INT
        2
    """
    _read_only: Dict[str, bool] = dict()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if keys := self.__annotations__.keys() & self._read_only:
            raise TypeError(f"Duplicated {keys=}")

        for k, v in self.__annotations__.items():
            variable = os.environ.get(k, '')
            if read_only := variable != '':
                if v is not str:
                    variable = eval(variable)
                    assert isinstance(variable, v)
                super().__setattr__(k, variable)
            self._read_only[k] = read_only

    def __getattr__(self, name: str) -> None:
        if name in self.__annotations__:
            return self.__annotations__[name]()
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in self.__annotations__ and self._read_only.get(name, False):
            logger.debug(f"Cannot set {name} to {value}.")
            return
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        variables = ' '.join(
            f'{k}={getattr(self, k)}' for k in self.__annotations__
        )
        return f"<{self.__name__} {variables}>"


class Store(metaclass=StoreMeta):
    CPU: bool

    CUDA: bool = torch.cuda.is_available()
    MPS: bool

    DRY_RUN: bool
    TRAIN_WITH_VAL_DATASET: bool


if parse(torch.__version__) >= parse('1.12'):
    import torch.backends.mps as mps
    if mps.is_available():
        Store.MPS = True

if not Store.CUDA and not Store.MPS:
    Store.CPU = True

assert Store.CPU ^ (Store.CUDA or Store.MPS)

if Store.CPU:
    Store.DRY_RUN = True
    Store.TRAIN_WITH_VAL_DATASET = True


class RegistryMeta(UserDict, NonInstantiableMeta):  # type: ignore[misc]
    """Meta class for registries.

    Under the hood, registries are simply dictionaries:

        >>> class Cat(metaclass=RegistryMeta): pass
        >>> class BritishShorthair: pass
        >>> Cat['british shorthair'] = BritishShorthair
        >>> Cat['british shorthair']
        <class '...BritishShorthair'>

    Users can also access registries via higher level APIs, i.e. `register`
    and `build`, for convenience.

    Registries can be subclassed.
    Derived classes of a registry are child registries:

        >>> class HairlessCat(Cat): pass
        >>> Cat.child('HairlessCat')
        <HairlessCat >
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize."""
        UserDict.__init__(self)
        NonInstantiableMeta.__init__(self, *args, **kwargs)

    def __call__(self, *args, **kwargs) -> NoReturn:
        """Prevent `Registry` classes from being initialized.

        Raises:
            TypeError: always.
        """
        raise TypeError("Registry cannot be initialized")

    def __missing__(self, key: str) -> NoReturn:
        """Missing key.

        Args:
            key: the missing key.

        Raises:
            KeyError: always.
        """
        logger.error(f"{key} does not exist in {self.__name__}")
        raise KeyError(key)

    def __repr__(self) -> str:
        items = ' '.join(f'{k}={v}' for k, v in self.items())
        return f"<{self.__name__} {items}>"

    def __setitem__(self, key: str, item, forced: bool = False) -> None:
        """Register ``item`` with name ``key``.

        Args:
            key: name to be registered as.
            item: object to be registered.
            forced: if set, ``item`` will always be registered.
                By default, ``item`` will only be registered if ``key`` is
                not registered yet.

        Raises:
            KeyError: if ``key`` is already registered and ``forced`` is not
                set.

        By default, registries refuse to alter the registered object, in order
        to prevent unintended name clashes:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> Cat['british shorthair'] = 'british shorthair'
            >>> Cat['british shorthair'] = 'BritishShorthair'
            Traceback (most recent call last):
                ...
            KeyError: 'british shorthair'

        Specify the ``forced`` option to force registration:

            >>> Cat.__setitem__(
            ...     'british shorthair',
            ...     'BritishShorthair',
            ...     forced=True,
            ... )
            >>> Cat['british shorthair']
            'BritishShorthair'
        """
        if not forced and key in self:
            logger.error(f"{key} already exist in {self.__name__}")
            raise KeyError(key)
        return super().__setitem__(key, item)

    def __subclasses__(
        self=...,  # type: ignore[assignment]
    ) -> List['RegistryMeta']:
        """Refer to `ABC subclassed by meta classes`_.

        Returns:
            Children registries.

        .. _ABC subclassed by meta classes:
           https://blog.csdn.net/LutingWang/article/details/128320057
        """
        if self is ...:
            return NonInstantiableMeta.__subclasses__(RegistryMeta) # type: ignore
        return super().__subclasses__() # type: ignore

    def child(self, key: str) -> 'RegistryMeta':
        """Get a direct or indirect derived child registry.

        Args:
            key: dot separated subclass names.

        Raises:
            ValueError: if zero or multiple children are found.

        Returns:
            The derived registry.
        """
        for child_name in key.split('.'):
            subclasses = tuple(
                subclass
                for subclass in self.__subclasses__()  # type: ignore[misc]
                if subclass.__name__ == child_name
            )
            if len(subclasses) == 0:
                raise ValueError(f"{key} is not a child of {self}")
            if len(subclasses) > 1:
                raise ValueError(f"{key} matches multiple children of {self}")
            self, = subclasses
        return self

    def _parse(self, key: str) -> Tuple['RegistryMeta', str]:
        """Parse the child name from the ``key``.

        Returns:
            The child registry and the name to be registered.
        """
        if '.' not in key:
            return self, key
        child_name, key = key.rsplit('.', 1)
        return self.child(child_name), key

    def register(
        self,
        keys: Union[Iterable[str], None] = None,
        **kwargs,
    ) -> Callable[[T], T]:
        """Register decorator.

        Args:
            keys: names to be registered as.
            kwargs: refer to `__setitem__`.

        Returns:
            Wrapper function.

        `register` is designed to be an decorator for classes and functions:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register()
            ... class Munchkin: pass
            >>> @Cat.register()
            ... def munchkin() -> str:
            ...     return 'munchkin'

        `register` has the following advantages:

        - default name

          By default, `register` uses the name of the registered object as
          ``keys``:

          >>> Cat['Munchkin']
          <class '...Munchkin'>
          >>> Cat['munchkin']
          <function munchkin at ...>

        - multiple names

          >>> @Cat.register(('British Longhair', 'british longhair'))
          ... class BritishLonghair: pass
          >>> 'British Longhair' in Cat
          True
          >>> 'british longhair' in Cat
          True

        - compatibility with child registries

          >>> class HairlessCat(Cat): pass
          >>> @Cat.register(('HairlessCat.CanadianHairless',))
          ... def canadian_hairless() -> str:
          ...     return 'canadian hairless'
          >>> HairlessCat
          <HairlessCat CanadianHairless=<function canadian_hairless at ...>>
        """

        def wrapper_func(obj: T) -> T:
            if keys is None:
                self.__setitem__(obj.__name__, obj, **kwargs)
            else:
                for key in keys:
                    registry, key = self._parse(key)
                    registry.__setitem__(key, obj, **kwargs)
            return obj

        return wrapper_func

    def _build(self, config: Config):
        """Build an instance according to the given config.

        Args:
            config: instance specification.

        Returns:
            The built instance.

        To customize the build process of instances, registries must overload
        `_build` with a class method:

            >>> class Cat(metaclass=RegistryMeta):
            ...     @classmethod
            ...     def _build(cls, config: Config) -> str:
            ...         obj = RegistryMeta._build(cls, config)
            ...         return obj.__class__.__name__.upper()
            >>> @Cat.register()
            ... class Munchkin: pass
            >>> Cat.build(Config(type='Munchkin'))
            'MUNCHKIN'
        """
        type_ = self[config.pop('type')]
        return type_(**config)

    def build(
        self,
        config: Config,
        default_config: Optional[Config] = None,
    ):
        """Call the registered object to construct a new instance.

        Args:
            config: build parameters.
            default_config: default configuration.

        Returns:
            The built instance.

        If the registered object is callable, the `build` method will
        automatically call the objects:

            >>> class Cat(metaclass=RegistryMeta): pass
            >>> @Cat.register()
            ... def tabby(name: str) -> str:
            ...     return f'Tabby {name}'
            >>> Cat.build(dict(type='tabby', name='Garfield'))
            'Tabby Garfield'

        Typically, ``config`` is a `Mapping` object.
        The ``type`` entry of ``config`` specifies the name of the registered
        object to be built.
        The other entries of ``config`` will be passed to the object's call
        method.

        ``default_config`` is the default configuration:

            >>> Cat.build(
            ...     dict(type='tabby'),
            ...     default_config=dict(name='Garfield'),
            ... )
            'Tabby Garfield'

        Refer to `_build` for customizations.
        """
        # TODO: remove default_config, use **kwargs
        default_config = (
            Config() if default_config is None else Config(default_config)
        )
        default_config.update(config)

        config = default_config.copy()
        registry, config.type = self._parse(config.type)

        try:
            return registry._build(config)
        except Exception as e:
            # config may be altered
            logger.error(f'Failed to build {default_config}')
            raise e


class Registry(metaclass=RegistryMeta):
    """Base registry.

    To create custom registry, inherit from the `Registry` class:

        >>> class CatRegistry(Registry): pass
    """


class NormRegistry(Registry):
    pass


def build_param_group(model: nn.Module, param_group: Config) -> Config:
    params = [
        p for n, p in model.named_parameters()
        if re.match(param_group['params'], n)
    ]
    param_group = param_group.copy()
    param_group.params = params
    return param_group


def build_param_groups(
    model: nn.Module,
    param_groups,
) -> List[Config]:
    if param_groups is None:
        return [Config(params=model.parameters())]
    return [build_param_group(model, param) for param in param_groups]


class OptimizerRegistry(Registry):

    @classmethod
    def _build(cls, config: Config) -> torch.optim.Optimizer:
        model = config.pop('model')
        config.params = build_param_groups(model, config.get('params', None))
        return RegistryMeta._build(cls, config)


class LrSchedulerRegistry(Registry):
    pass


NormRegistry['BN1d'] = nn.BatchNorm1d
NormRegistry['BN2d'] = NormRegistry['BN'] = nn.BatchNorm2d
NormRegistry['BN3d'] = nn.BatchNorm3d
NormRegistry['SyncBN'] = nn.SyncBatchNorm
NormRegistry['GN'] = nn.GroupNorm
NormRegistry['LN'] = nn.LayerNorm
NormRegistry['IN1d'] = nn.InstanceNorm1d
NormRegistry['IN2d'] = NormRegistry['IN'] = nn.InstanceNorm2d
NormRegistry['IN3d'] = nn.InstanceNorm3d

for _, class_ in inspect.getmembers(torch.optim, inspect.isclass):
    assert issubclass(class_, torch.optim.Optimizer)
    OptimizerRegistry.register()(class_)

for _, class_ in inspect.getmembers(torch.optim.lr_scheduler, inspect.isclass):
    if issubclass(class_, torch.optim.lr_scheduler._LRScheduler):
        LrSchedulerRegistry.register()(class_)

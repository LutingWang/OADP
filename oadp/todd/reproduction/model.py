__all__ = [
    'Finder',
    'ParameterFinder',
    'ModuleFinder',
    'FrozenMixin',
    'state_dict_hook',
]

import re
import types
from typing import Any, Generic, Iterable, MutableMapping, TypeVar
from typing_extensions import Self

import torch.nn as nn

from ..base import Config, get_

T = TypeVar('T')


class Finder(Generic[T]):
    """Find objects in a model by names, regex, type, or types.

    Consider the following model:

        >>> class Model(nn.Module):
        ...     def __init__(self, *args, **kwargs) -> None:
        ...         super().__init__(*args, **kwargs)
        ...         self._conv = nn.Conv2d(3, 8, 3)
        ...         self._bn = nn.BatchNorm2d(8)
        ...         self._conv1 = nn.Conv2d(8, 16, 3)
        ...         self._bn1 = nn.BatchNorm2d(16)
        ...         self._relu = nn.ReLU()
        >>> model = Model().requires_grad_(False)

    We can initialize a Finder as follows:

        >>> finder = Finder(model)

    Suppose we want to find the first convolution and batch normalization
    layers, this is how we can do it:

        >>> finder.find_by_names(['._conv', '._bn'])
        [Conv2d(3, 8, ...), BatchNorm2d(8, ...)]

    To find all convolution layers, we can do:

        >>> finder.find_by_regex('_conv.*', 'modules')
        [Conv2d(3, 8, ...), Conv2d(8, 16, ...)]

    `find_by_type` provides another way to find convolution layers:

        >>> finder.find_by_type(nn.Conv2d, 'modules')
        [Conv2d(3, 8, ...), Conv2d(8, 16, ...)]

    Different from `find_by_type`, `find_by_types` can match multiple types:

        >>> finder.find_by_types([nn.Conv2d, nn.BatchNorm2d], 'modules')
        [Conv2d(3, 8, ...), BatchNorm2d(8, ...), Conv2d(8, 16, ...), BatchNorm\
2d(16, ...)]

    `find_by_configs` is a more flexible way to find objects.
    For example, we can find the first convolution and batch normalization
    layers by:

        >>> finder.find_by_config(Config(names=['._conv', '._bn']))
        [Conv2d(3, 8, ...), BatchNorm2d(8, ...)]
    """

    def __init__(self, model: nn.Module) -> None:
        """Initialize a Finder.

        Args:
            model: a model to find objects in.
        """
        self._model = model

    def _formulate(self, objects):
        """Formulate objects into desirable types.

        Args:
            objects: objects to formulate.

        Returns:
            A list of formulated objects.

        This method is meant to be overridden by subclasses.
        """
        return objects

    def find_by_names(self, names: Iterable[str]):
        """Find objects by exact name match.

        Args:
            names: names of objects to find.

        Returns:
            A list of objects.

        Refer to `todd.get_` for more details about how names are resolved.
        """
        objects = [get_(self._model, name) for name in names]
        return self._formulate(objects)

    def find_by_regex(self, regex: str, member: str):
        """Find objects by regex match.

        Args:
            regex: a regex to match names.
            member: type of member to search in.

        Returns:
            A list of objects.

        Compared with `find_by_names`, this method is more flexible, but it
        requires traversing the model, which is slower.
        """
        objects = [
            object_
            for name, object_ in getattr(self._model, f'named_{member}')()
            if re.search(regex, name)
        ]
        return self._formulate(objects)

    def find_by_type(
        self,
        type_,
        member: str,
    ):
        """Find objects by type.

        Args:
            type_: a type or a tuple of types to match.
            member: type of member to search in.

        Returns:
            A list of objects.
        """
        objects = [
            object_ for object_ in getattr(self._model, member)()
            if isinstance(object_, type_)
        ]
        return self._formulate(objects)

    def find_by_types(
        self,
        types: Iterable[type],
        *args,
        **kwargs,
    ):
        """Find objects by types.

        Args:
            types: types to match.

        Returns:
            A list of objects.
        """
        return self.find_by_type(tuple(types), *args, **kwargs)

    def find_by_config(self, config: Config):
        """Find objects by config.

        Args:
            config: the match config.

        Returns:
            A list of objects.

        Raises:
            ValueError: if ``config`` has no valid key.
            ValueError: if ``config`` has multiple valid keys.

        Valid keys include ``names``, ``regex``, ``type_``, and ``types``.
        No valid key or multiple keys will raise an error:

            >>> model = nn.Linear(1, 1)
            >>> finder = Finder(model)
            >>> finder.find_by_config(dict(name='.weight'))
            Traceback (most recent call last):
                ...
            ValueError: no valid key in config={'name': '.weight'}
            >>> finder.find_by_config(dict(names=['.weight'], regex='.*'))
            Traceback (most recent call last):
                ...
            ValueError: multiple keys in config={'names': ['.weight'], 'regex'\
: '.*'}
        """
        keys = config.keys() & ['names', 'regex', 'type_', 'types']
        if len(keys) == 0:
            raise ValueError(f"no valid key in {config=}")
        if len(keys) > 1:
            raise ValueError(f"multiple keys in {config=}")
        key = keys.pop()
        return getattr(self, f'find_by_{key}')(**config)

    def determine_modes(
        self,
        configs: Iterable[Config],
    ):
        """Determine modes of objects.

        Args:
            configs: configs specifying modes.

        Returns:
            A dict mapping objects to modes.

        Example:
            >>> class Model(nn.Module):
            ...     def __init__(self, *args, **kwargs) -> None:
            ...         super().__init__(*args, **kwargs)
            ...         self.conv = nn.Conv1d(1, 2, 3)
            ...         self.bn = nn.BatchNorm1d(3)
            >>> model = Model().requires_grad_(False)
            >>> finder = Finder(model)
            >>> finder.determine_modes([
            ...     Config(names=['.conv'], mode=False),
            ...     Config(regex='bias', member='parameters', mode=True),
            ...     Config(names=['.bn.bias'], mode=Ellipsis),
            ... ])
            {Conv1d(1, 2, kernel_size=(3,), stride=(1,)): False, Parameter con\
taining:
            tensor([..., ...]): True, Parameter containing:
            tensor([0., 0., 0.]): Ellipsis}
        """
        modes = dict()
        for config in configs:
            config = config.copy()
            mode = config.pop('mode')
            for object_ in self.find_by_config(config):
                modes[object_] = mode
        return modes


class ParameterFinder(Finder[nn.Parameter]): # type: ignore
    """Find parameters."""

    def _formulate(self, objects):
        """Formulate objects into parameters.

        Args:
            objects: objects to formulate.

        Raises:
            TypeError: if any element of ``objects`` is not supported.

        Returns:
            A list of parameters.

        Supported elements include `nn.Parameter` and `nn.Module`:

            >>> import torch
            >>> model = nn.Linear(1, 2).requires_grad_(False)
            >>> _ = model.weight.data.fill_(3.0)
            >>> _ = model.bias.data.fill_(4.0)
            >>> finder = ParameterFinder(model)
            >>> finder._formulate([
            ...     nn.Parameter(torch.tensor(5.0), requires_grad=False),
            ...     model,
            ... ])
            [Parameter containing:
            tensor(5.), Parameter containing:
            tensor([[3.],
                    [3.]]), Parameter containing:
            tensor([4., 4.])]

        Other types triggers exception:

            >>> finder._formulate([torch.tensor(1.0)])
            Traceback (most recent call last):
               ...
            TypeError: unsupported object_=tensor(1.)
        """
        parameters = []
        for object_ in objects:
            if isinstance(object_, nn.Parameter): # type: ignore
                parameters.append(object_)
            elif isinstance(object_, nn.Module):
                parameters.extend(object_.parameters())
            else:
                raise TypeError(f"unsupported {object_=}")
        return parameters


class ModuleFinder(Finder[nn.Module]):
    """Find modules."""

    def _formulate(self, objects):
        """Formulate objects into modules.

        Args:
            objects: objects to formulate.

        Returns:
            A list of modules.

        Raises:
            TypeError: if any element of ``objects`` is not `nn.Module`.

        In case users accidentally pass in `nn.Parameter`, a `TypeError` will
        be raised:

            >>> model = nn.Linear(2, 1).requires_grad_(False)
            >>> finder = ModuleFinder(model)
            >>> finder._formulate([model.weight])
            Traceback (most recent call last):
                ...
            TypeError: unsupported module=Parameter containing:
            tensor([[..., ...]])
        """
        modules = list(objects)
        for module in modules:
            if not isinstance(module, nn.Module):
                raise TypeError(f"unsupported {module=}")
        return modules


class FrozenMixin(nn.Module):
    """Freeze parameters."""

    def __init__(
        self,
        requires_grad_configs: Iterable[Config] = tuple(),
        train_configs: Iterable[Config] = tuple(),
        with_state_dict_hook: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._requires_grad_configs = tuple(requires_grad_configs)
        self._train_configs = tuple(train_configs)
        self._with_state_dict_hook = with_state_dict_hook
        if with_state_dict_hook:
            self._register_state_dict_hook(state_dict_hook)

    @property
    def requires_grad_configs(self):
        return self._requires_grad_configs

    @property
    def train_configs(self):
        return self._train_configs

    def requires_grad_(self, requires_grad: bool = True) -> Self:
        self = super().requires_grad_(requires_grad)
        finder = ParameterFinder(self)
        configs = self._requires_grad_configs
        modes = {
            parameter: mode
            for parameter, mode in finder.determine_modes(configs).items()
            if mode is not ...
        }
        for parameter, mode in modes.items():
            parameter.requires_grad_(mode)
        return self

    def train(self, mode: bool = True) -> Self:
        self = super().train(mode)
        finder = ModuleFinder(self)
        configs = self._train_configs
        modes = {
            module: module.training if mode is ... else mode
            for module, mode in finder.determine_modes(configs).items()
        }
        for module in modes:
            module.train(modes[module])
        return self


def state_dict_hook(
    model: FrozenMixin,
    state_dict: MutableMapping[str, Any],
    prefix: str,
    *args,
    **kwargs,
) -> None:
    """Hook for `torch.nn.Module.load_state_dict`.

    Args:
        model: the model to load state dict.
        state_dict: the state dict to load.
        prefix: the prefix of the model.
        *args: other args.
        **kwargs: other kwargs.

    Example:

        >>> class Model(FrozenMixin):
        ...     def __init__(self, *args, **kwargs) -> None:
        ...         super().__init__(
        ...             *args,
        ...             requires_grad_configs=[
        ...                 Config(names=['.conv'], mode=False),
        ...                 Config(names=['.bn'], mode=...),
        ...             ],
        ...             with_state_dict_hook=True,
        ...             **kwargs,
        ...         )
        ...         self.conv = nn.Conv1d(1, 2, 1)
        ...         self.bn = nn.BatchNorm1d(2)
        >>> Model().state_dict()
        OrderedDict([('bn.weight', tensor([1., 1.])), ('bn.bias', tensor([0., \
0.])), ('bn.running_mean', tensor([0., 0.])), ('bn.running_var', tensor([1., 1\
.])), ('bn.num_batches_tracked', tensor(0))])
        >>> nn.Sequential(Model()).state_dict()
        OrderedDict([('0.bn.weight', tensor([1., 1.])), ('0.bn.bias', tensor([\
0., 0.])), ('0.bn.running_mean', tensor([0., 0.])), ('0.bn.running_var', tenso\
r([1., 1.])), ('0.bn.num_batches_tracked', tensor(0))])
    """
    if len(state_dict) == 0:
        return
    finder = ParameterFinder(model)
    modes = finder.determine_modes(model.requires_grad_configs)
    parameter_names = {
        parameter: name
        for name, parameter in model.named_parameters()
    }
    for parameter, mode in modes.items():
        if mode is False:
            name = parameter_names[parameter]
            name = name.replace('_fsdp_wrapped_module.', '')
            state_dict.pop(prefix + name)

__all__ = [
    'StatusMixin',
]

import enum
import functools
from typing import Any, Callable, Generic, Sequence, TypeVar, no_type_check

T = TypeVar('T', bound=enum.Enum)


class StatusMixin(Generic[T]):

    def __init__(self, status: T) -> None:
        self._status = status

    @property
    def status(self) -> T:
        return self._status

    @staticmethod
    def transit(
        source,
        target,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        if not isinstance(source, Sequence):
            assert not isinstance(target, Sequence)
            sources = [source]
            targets = [target]
        elif not isinstance(target, Sequence):
            sources = list(source)
            targets = [target] * len(source)
        else:
            sources = list(source)
            targets = list(target)
        assert len(sources) == len(targets) == len(set(sources)), \
            (source, target)

        @no_type_check
        def wrapper(wrapped_func):

            @functools.wraps(wrapped_func)
            def wrapper_func(self: StatusMixin, *args, **kwargs):
                if self._status not in sources:
                    raise RuntimeError(f"{self} is not in {sources}.")
                self._status = targets[sources.index(self._status)]
                return wrapped_func(self, *args, **kwargs)

            return wrapper_func

        return wrapper

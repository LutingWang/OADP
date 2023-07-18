import string

from .base import AdaptRegistry, BaseAdapt


@AdaptRegistry.register()
class Custom(BaseAdapt):
    """Custom adaptation described using patterns.

    Examples:
        >>> adapt = Custom(pattern='a + b * c')
        >>> adapt(1, 2, 3)
        7
        >>> adapt = Custom(pattern='a + var1 * var2')
        >>> adapt(1, var1=2, var2=3)
        7
        >>> adapt = Custom(pattern='a + b * c')
        >>> adapt(1, 2, 3, b=4)
        Traceback (most recent call last):
            ...
        RuntimeError: {'b'}
    """

    def __init__(self, *args, pattern: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._pattern = pattern

    def forward(self, *args, **kwargs):
        locals_ = dict(zip(string.ascii_letters, args))
        if len(locals_.keys() & kwargs.keys()) != 0:
            raise RuntimeError(locals_.keys() & kwargs.keys())
        locals_.update(kwargs)
        return eval(self._pattern, None, locals_)

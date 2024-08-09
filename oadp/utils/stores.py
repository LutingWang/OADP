__all__ = [
    'Store',
]

import todd


class Store(metaclass=todd.utils.StoreMeta):
    DUMP: str

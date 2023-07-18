__all__ = [
    'SGR',
]

import enum
from typing import Iterable
from typing_extensions import Self


class SGR(enum.IntEnum):
    """Select Graphic Rendition.

    Refer to https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    NORMAL = 0
    BOLD = enum.auto()
    FAINT = enum.auto()
    ITALIC = enum.auto()
    UNDERLINE = enum.auto()
    BLINK_SLOW = enum.auto()
    BLINK_FAST = enum.auto()
    REVERSE = enum.auto()
    CONCEAL = enum.auto()
    CROSSED_OUT = enum.auto()

    FG_BLACK = 30
    FG_RED = enum.auto()
    FG_GREEN = enum.auto()
    FG_YELLOW = enum.auto()
    FG_BLUE = enum.auto()
    FG_MAGENTA = enum.auto()
    FG_CYAN = enum.auto()
    FG_WHITE = enum.auto()

    BG_BLACK = 40
    BG_RED = enum.auto()
    BG_GREEN = enum.auto()
    BG_YELLOW = enum.auto()
    BG_BLUE = enum.auto()
    BG_MAGENTA = enum.auto()
    BG_CYAN = enum.auto()
    BG_WHITE = enum.auto()

    @classmethod
    def CSI(cls, parameters: Iterable[Self]) -> str:
        """Control Sequence Introducer."""
        return f'\033[{";".join(str(p.value) for p in parameters)}m'

    @classmethod
    def format(cls, text: str, *args: Self) -> str:
        return cls.CSI(args) + text + cls.CSI(tuple())


if __name__ == '__main__':
    import itertools

    for fg, bg in itertools.product(range(30, 38), range(40, 48)):
        text = ' '.join(
            SGR.format(f'{effect}:{fg};{bg}', SGR(effect), SGR(fg), SGR(bg))
            for effect in range(10)
        )
        print(text)

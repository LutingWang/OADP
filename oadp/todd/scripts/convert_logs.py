"""Convert Logs.

Example:
    $ python -m todd.scripts.convert_logs \\
    > --reader \\
    >   type:\\'JSONReader\\' \\
    >   filepath:\\'log.json\\' \\
    > --selector \\
    >   type:\\'ContainsSelector\\' \\
    >   cond:\\'mode\\' \\
    > --selector \\
    >   type:\\'CustomSelector\\' \\
    >   cond:\\'mode==\\"train\\"\\' \\
    > --writer \\
    >   type:\\'TensorBoardWriter\\' \\
    >   filepath:\\'work_dirs/tb\\' \\
    >   main_tag:\\'mode\\' \\
    >   tag_value_dict:dict\\(loss=\\'loss\\'\\) \\
    >   global_step:\\'\\(epoch\\-1\\)\\*2\\+iter\\' \\
    >   walltime:\\'time\\'
"""

import argparse
import itertools
import json
import numbers
import os
import time
from abc import ABC, abstractmethod
from typing import Generator

from ..todd.base import DictAction, Registry


class BaseReader(ABC):

    def __init__(self, filepath: str) -> None:
        super().__init__()
        self._file = open(filepath)

    def __del__(self) -> None:
        self._file.close()

    def _readline(self) -> str:
        line = self._file.readline()
        if len(line) == 0:
            raise StopIteration
        return line

    @abstractmethod
    def read(self) -> dict:
        pass

    def __iter__(self) -> Generator[dict, None, None]:
        while True:
            try:
                yield self.read()
            except StopIteration:
                break


class ReaderRegistry(Registry):
    pass


@ReaderRegistry.register()
class TxtReader(BaseReader):

    def read(self) -> dict:
        raise NotImplementedError


@ReaderRegistry.register()
class JSONReader(BaseReader):

    def read(self) -> dict:
        line = self._readline()
        return json.loads(line)


class BaseSelector(ABC):

    def __init__(self, cond: str) -> None:
        self._cond = cond

    @abstractmethod
    def filter(self, log: dict) -> bool:
        pass

    def __call__(self, log: dict) -> bool:
        return self.filter(log)


class SelectorRegistry(Registry):
    pass


@SelectorRegistry.register()
class ContainsSelector(BaseSelector):

    def filter(self, log: dict) -> bool:
        return self._cond in log


@SelectorRegistry.register()
class CustomSelector(BaseSelector):

    def filter(self, log: dict) -> bool:
        return eval(self._cond, None, log)


class BaseWriter(ABC):

    def __init__(
        self,
        filepath: str,
    ) -> None:
        self._filepath = filepath

    @abstractmethod
    def write(self, log: dict) -> None:
        pass

    def close(self) -> None:
        pass

    def __call__(self, log: dict) -> None:
        self.write(log)


class WriterRegistry(Registry):
    pass


@WriterRegistry.register()
class TensorBoardWriter(BaseWriter):

    def __init__(
        self,
        *args,
        main_tag,
        tag_value_dict,
        global_step,
        walltime,
        **kwargs,
    ) -> None:
        from torch.utils.tensorboard.writer import SummaryWriter

        super().__init__(*args, **kwargs)
        self._main_tag = main_tag
        self._tag_value_dict = tag_value_dict
        self._global_step = global_step
        self._walltime = walltime

        self._timer = 0

        self._filepath = os.path.join(
            self._filepath,
            time.strftime('%Y%m%d_%H%M%S', time.localtime()) + '.tensorboard',
        )
        self._writer = SummaryWriter(self._filepath)

    def write(self, log: dict) -> None:
        main_tag = None if self._main_tag is None else eval(
            self._main_tag, None, log
        )
        tag_value_dict = {  # yapf: disable
            tag: eval(value, None, log)
            for tag, value in self._tag_value_dict.items()
        }
        global_step = None if self._global_step is None else eval(
            self._global_step, None, log
        )
        walltime = None if self._walltime is None else eval(
            self._walltime, None, log
        )

        self._timer += walltime or global_step or 1
        for tag, value in tag_value_dict.items():
            if main_tag is not None:
                tag = f'{main_tag}/{tag}'
            if isinstance(value, str):
                self._writer.add_text(tag, value, global_step, self._timer)
            elif isinstance(value, numbers.Number):
                self._writer.add_scalar(tag, value, global_step, self._timer)
            else:
                raise TypeError(
                    f"Unsupported data {value} with type {type(value)}"
                )

    def close(self) -> None:
        self._writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.replace('\n    ', '\n  '),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('--reader', nargs='+', action=DictAction)
    parser.add_argument('--selector', nargs='+', action=DictAction)
    parser.add_argument('--writer', nargs='+', action=DictAction)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    readers = [ReaderRegistry.build(reader) for reader in args.reader]
    selectors = [
        SelectorRegistry.build(selector) for selector in args.selector
    ]
    writers = [WriterRegistry.build(writer) for writer in args.writer]

    for log in itertools.chain(*readers):
        if all(selector(log) for selector in selectors):
            for writer in writers:
                writer(log)
    for writer in writers:
        writer.close()


if __name__ == '__main__':
    main()

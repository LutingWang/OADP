import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import torch

from .base import (
    AccessLayerRegistry,
    BaseAccessLayer,
    BaseDataset,
    Codec,
    DatasetRegistry,
)

# TODO: update


@AccessLayerRegistry.register()
class ZipAccessLayer(BaseAccessLayer[str]):

    def _init(self):
        assert self._readonly, 'ZipAccessLayer is read-only.'
        assert self._codec is Codec.PYTORCH, \
            'ZipAccessLayer only supports PyTorch.'
        data_root_parts = Path(self._data_root).parts
        zip_indices = [  # yapf: disable
            i for i, part in enumerate(data_root_parts)
            if part.endswith('.zip')
        ]
        if len(zip_indices) == 0:
            raise ValueError(f'{self._data_root} does not contain a zip file.')
        elif len(zip_indices) > 1:
            raise ValueError(f'{self._data_root} contains multiple zip files.')
        zip_index = zip_indices[0]
        self._zip_root = zipfile.Path(
            os.path.join(*data_root_parts[:zip_index + 1]),
            os.path.join(
                *data_root_parts[zip_index + 1:],
                self._task_name,
                '',
            ),
        )

    @property
    def exists(self) -> bool:
        return self._zip_root.exists()

    def touch(self):
        raise NotImplementedError

    def __iter__(self) -> Iterator[str]:
        return (path.name for path in self._zip_root.iterdir())

    def __len__(self) -> int:
        return len(list(self._zip_root.iterdir()))

    def __getitem__(self, key: str) -> Any:
        data_file = self._zip_root / key
        bytes_io = BytesIO(data_file.read_bytes())
        return torch.load(bytes_io, map_location='cpu')

    def __setitem__(self, key: str, value: Any):
        raise NotImplementedError

    def __delitem__(self, key: str):
        raise NotImplementedError


@DatasetRegistry.register()
class ZipDataset(BaseDataset[str]):
    ACCESS_LAYER = ZipAccessLayer

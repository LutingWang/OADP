__all__ = [
    'AccessLayer',
    'COCOAccessLayer',
    'LVISAccessLayer',
]

from abc import ABC, abstractmethod
from typing import Literal, Never, TypeVar

import torch
from todd.datasets.access_layers import BaseAccessLayer, PthAccessLayer

from oadp.oake.blocks.runners import Output as BlockOutput
from oadp.oake.objects.runners import Output as ObjectOutput

from .registries import DPAccessLayerRegistry

T = TypeVar('T')


class BaseMixin(PthAccessLayer[T], ABC):
    TASK_NAME: str

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, task_name=self.TASK_NAME, **kwargs)


class BaseGlobalAccessLayer(
    BaseMixin[torch.Tensor],
    PthAccessLayer[torch.Tensor],
):
    pass


class BaseBlockAccessLayer(
    BaseMixin[BlockOutput],
    PthAccessLayer[BlockOutput],
):
    pass


class BaseObjectAccessLayer(
    BaseMixin[ObjectOutput],
    PthAccessLayer[ObjectOutput],
):

    def __getitem__(self, key: str) -> ObjectOutput:
        item = super().__getitem__(key)
        bboxes = item['bboxes']
        indices = bboxes.indices(min_wh=(4, 4))
        return ObjectOutput(
            tensors=item['tensors'][indices],
            bboxes=bboxes[indices],
            categories=item['categories'][indices],
        )


class AccessLayer(
    BaseAccessLayer[T, tuple[torch.Tensor, BlockOutput, ObjectOutput]],
    ABC,
):
    DATA_ROOT = 'work_dirs/oake'

    def __init__(
        self,
        *args,
        global_: BaseGlobalAccessLayer,
        block: BaseBlockAccessLayer,
        object_: BaseObjectAccessLayer,
        **kwargs,
    ) -> None:
        super().__init__(self.DATA_ROOT, *args, **kwargs)
        self._global = global_
        self._block = block
        self._object = object_

    def __getitem__(
        self,
        key: T,
    ) -> tuple[torch.Tensor, BlockOutput, ObjectOutput]:
        return self._global[key], self._block[key], self._object[key]

    def __setitem__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __delitem__(self, *args, **kwargs) -> Never:
        raise NotImplementedError

    def __iter__(self) -> Never:
        raise NotImplementedError

    def __len__(self) -> Never:
        raise NotImplementedError

    @property
    def exists(self) -> Never:
        raise NotImplementedError

    def touch(self) -> Never:
        raise NotImplementedError


class COCOGlobalAccessLayer(BaseGlobalAccessLayer):
    TASK_NAME = 'coco_globals_cuda_train/output'


class COCOBlockAccessLayer(BaseBlockAccessLayer):
    TASK_NAME = 'coco_blocks_cuda_train/output'


class COCOObjectAccessLayer(BaseObjectAccessLayer):
    TASK_NAME = 'coco_objects_cuda_train/output'


@DPAccessLayerRegistry.register_()
class COCOAccessLayer(AccessLayer[int]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            global_=COCOGlobalAccessLayer(self.DATA_ROOT),
            block=COCOBlockAccessLayer(self.DATA_ROOT),
            object_=COCOObjectAccessLayer(self.DATA_ROOT),
            **kwargs,
        )

    def __getitem__(self, key: int) -> T:
        return super().__getitem__(f'{key:012d}')


class LVISMixin(PthAccessLayer[T], ABC):

    @abstractmethod
    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        pass

    def __getitem__(self, key: str) -> T:
        split, key = key.split('/')
        key = self.get_key(split, key)
        return super().__getitem__(key)


class LVISGlobalAccessLayer(LVISMixin, BaseGlobalAccessLayer):
    TASK_NAME = ''

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco_globals_cuda_{split}/output/{key}'


class LVISBlockAccessLayer(LVISMixin, BaseBlockAccessLayer):
    TASK_NAME = ''

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco_blocks_cuda_{split}/output/{key}'


class LVISObjectAccessLayer(LVISMixin, BaseObjectAccessLayer):
    TASK_NAME = 'lvis_objects_cuda_train/output'

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'{split}2017_{key}'


@DPAccessLayerRegistry.register_()
class LVISAccessLayer(AccessLayer[str]):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            global_=LVISGlobalAccessLayer(self.DATA_ROOT),
            block=LVISBlockAccessLayer(self.DATA_ROOT),
            object_=LVISObjectAccessLayer(self.DATA_ROOT),
            **kwargs,
        )

    def __getitem__(self, key: str) -> T:
        prefix = 'data/lvis/'
        suffix = '.jpg'
        assert key.startswith(prefix) and key.endswith(suffix)
        key = key.removeprefix(prefix).removesuffix(suffix)

        split, key = key.split('/')
        assert split in ['train2017', 'val2017']
        split = split.removesuffix('2017')

        key = f'{split}/{key}'
        return super().__getitem__(key)

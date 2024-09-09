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


class BaseGlobalAccessLayer(PthAccessLayer[torch.Tensor]):
    pass


class BaseBlockAccessLayer(PthAccessLayer[BlockOutput]):
    pass


class BaseObjectAccessLayer(PthAccessLayer[ObjectOutput]):

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

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_globals_cuda_train/output',
            **kwargs,
        )


class COCOBlockAccessLayer(BaseBlockAccessLayer):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_blocks_cuda_train/output',
            **kwargs,
        )


class COCOObjectAccessLayer(BaseObjectAccessLayer):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_objects_cuda_train/output',
            **kwargs,
        )


@DPAccessLayerRegistry.register_()
class COCOAccessLayer(AccessLayer[int]):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            global_=COCOGlobalAccessLayer(self.DATA_ROOT, model=model),
            block=COCOBlockAccessLayer(self.DATA_ROOT, model=model),
            object_=COCOObjectAccessLayer(self.DATA_ROOT, model=model),
            **kwargs,
        )

    def __getitem__(self, key: int) -> T:
        return super().__getitem__(f'{key:012d}')


class LVISMixin(PthAccessLayer[T], ABC):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model

    @abstractmethod
    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        pass

    def __getitem__(self, key: str) -> T:
        split, key = key.split('/')
        key = self.get_key(split, key)
        return super().__getitem__(key)


class LVISGlobalAccessLayer(LVISMixin, BaseGlobalAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco/{self._model}_globals_cuda_{split}/output/{key}'


class LVISBlockAccessLayer(LVISMixin, BaseBlockAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco/{self._model}_blocks_cuda_{split}/output/{key}'


class LVISObjectAccessLayer(LVISMixin, BaseObjectAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return (
            f'lvis/{self._model}_objects_cuda_train/output/{split}2017_{key}'
        )


@DPAccessLayerRegistry.register_()
class LVISAccessLayer(AccessLayer[str]):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            global_=LVISGlobalAccessLayer(self.DATA_ROOT, model=model),
            block=LVISBlockAccessLayer(self.DATA_ROOT, model=model),
            object_=LVISObjectAccessLayer(self.DATA_ROOT, model=model),
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

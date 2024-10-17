__all__ = [
    'BaseGlobalAccessLayer',
    'BaseBlockAccessLayer',
    'BaseObjectAccessLayer',
    'COCOGlobalAccessLayer',
    'COCOBlockAccessLayer',
    'COCOObjectAccessLayer',
    'LVISGlobalAccessLayer',
    'LVISBlockAccessLayer',
    'LVISObjectAccessLayer',
    'Objects365GlobalAccessLayer',
    'Objects365BlockAccessLayer',
    'Objects365ObjectAccessLayer',
]

from abc import ABC, abstractmethod
from typing import Literal, TypeVar

import torch
from todd.datasets.access_layers import PthAccessLayer

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


@DPAccessLayerRegistry.register_()
class COCOGlobalAccessLayer(BaseGlobalAccessLayer):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_globals_cuda_train/output',
            **kwargs,
        )


@DPAccessLayerRegistry.register_()
class COCOBlockAccessLayer(BaseBlockAccessLayer):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_blocks_cuda_train/output',
            **kwargs,
        )


@DPAccessLayerRegistry.register_()
class COCOObjectAccessLayer(BaseObjectAccessLayer):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(
            *args,
            task_name=f'coco/{model}_objects_cuda_train/output',
            **kwargs,
        )


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


@DPAccessLayerRegistry.register_()
class LVISGlobalAccessLayer(LVISMixin, BaseGlobalAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco/{self._model}_globals_cuda_{split}/output/{key}'


@DPAccessLayerRegistry.register_()
class LVISBlockAccessLayer(LVISMixin, BaseBlockAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'coco/{self._model}_blocks_cuda_{split}/output/{key}'


@DPAccessLayerRegistry.register_()
class LVISObjectAccessLayer(LVISMixin, BaseObjectAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return (
            f'lvis/{self._model}_objects_cuda_train/output/{split}2017_{key}'
        )

@DPAccessLayerRegistry.register_()
class Objects365GlobalAccessLayer(LVISMixin, BaseGlobalAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'objects365/{self._model}_globals_cuda_{split}/output/{key}'


@DPAccessLayerRegistry.register_()
class Objects365BlockAccessLayer(LVISMixin, BaseBlockAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return f'objects365/{self._model}_blocks_cuda_{split}/output/{key}'


@DPAccessLayerRegistry.register_()
class Objects365ObjectAccessLayer(LVISMixin, BaseObjectAccessLayer):

    def get_key(self, split: Literal['train', 'val'], key: str) -> str:
        return (
            f'objects365/{self._model}_objects_cuda_{split}/output/{key}'
        )

class V3DetMixin(PthAccessLayer[T], ABC):

    def __init__(self, *args, model: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = model

    @abstractmethod
    def get_key(self, key: str) -> str:
        pass

    def __getitem__(self, key: str) -> T:
        key = self.get_key(key)
        return super().__getitem__(key)
    
@DPAccessLayerRegistry.register_()
class V3DetGlobalAccessLayer(V3DetMixin, BaseGlobalAccessLayer):

    def get_key(self, key: str) -> str:
        return f'v3det/{self._model}_globals_cuda/output/{key}'


@DPAccessLayerRegistry.register_()
class V3DetBlockAccessLayer(V3DetMixin, BaseBlockAccessLayer):

    def get_key(self, key: str) -> str:
        return f'v3det/{self._model}_blocks_cuda/output/{key}'


@DPAccessLayerRegistry.register_()
class V3DetObjectAccessLayer(V3DetMixin, BaseObjectAccessLayer):

    def get_key(self, key: str) -> str:
        return (
            f'v3det/{self._model}_objects_cuda/output/{key}'
        )
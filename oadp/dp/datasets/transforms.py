__all__ = [
    'LoadOAKEMixin',
    'LoadOAKEGlobal',
    'LoadOAKEBlock',
    'LoadOAKEObject',
    'AssignOAKEBlockLabels',
]
import os.path as osp
import enum
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, cast

import todd
import todd.tasks.object_detection as od
import torch
from mmdet.datasets.transforms import PackDetInputs
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes
from todd.bases.registries import BuildPreHookMixin, Item, RegistryMeta
from todd.datasets.access_layers import PthAccessLayer

from oadp.oake.blocks.runners import Output as BlockOutput
from oadp.oake.objects.runners import Output as ObjectOutput
from oadp.utils import Globals

from ..registries import DPTransformRegistry
from .registries import DPAccessLayerRegistry

T = TypeVar('T')


class LoadOAKEMixin(Generic[T], BuildPreHookMixin, ABC):

    def __init__(
        self,
        *args,
        access_layer: PthAccessLayer[T],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._access_layer = access_layer

    @classmethod
    def access_layer_build_pre_hook(
        cls,
        config: todd.Config,
        registry: RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config.access_layer = DPAccessLayerRegistry.build_or_return(
            config.access_layer,
            data_root='work_dirs/oake',
        )
        return config

    @classmethod
    def build_pre_hook(
        cls,
        config: todd.Config,
        registry: RegistryMeta,
        item: Item,
    ) -> todd.Config:
        config = cls.access_layer_build_pre_hook(config, registry, item)
        return config

    @abstractmethod
    def _load(self, item: T) -> dict[str, Any]:
        pass

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        image_id = inputs['image_id']
        item = self._access_layer[image_id]
        oake = self._load(item)
        inputs.update(oake)
        return inputs


@DPTransformRegistry.register_()
class LoadOAKEGlobal(LoadOAKEMixin[torch.Tensor]):

    def _load(self, item: torch.Tensor) -> dict[str, Any]:
        return dict(clip_global=item)


@DPTransformRegistry.register_()
class LoadOAKEBlock(LoadOAKEMixin[BlockOutput]):

    def _load(self, item: BlockOutput) -> dict[str, Any]:
        block_bboxes = item['bboxes']  # od.FlattenBBoxesXYXY
        return dict(
            block_bboxes=block_bboxes,
            clip_blocks=item['embeddings'],
        )


@DPTransformRegistry.register_()
class LoadOAKEObject(LoadOAKEMixin[ObjectOutput]):

    def _load(self, item: ObjectOutput) -> dict[str, Any]:
        object_bboxes = item['bboxes'].to(od.FlattenBBoxesXYXY)
        return dict(
            object_bboxes=object_bboxes,
            clip_objects=item['tensors'],
        )


@DPTransformRegistry.register_()
class AssignOAKEBlockLabels:

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        gt_bboxes: od.FlattenBBoxesXYXY = inputs['gt_bboxes']
        gt_labels: torch.Tensor = inputs['gt_labels']
        block_bboxes: od.FlattenBBoxesXYXY = inputs['block_bboxes']

        num_categories = Globals.categories.num_all
        indices = gt_labels < num_categories  # filter out pseudo labels
        gt_bboxes = gt_bboxes[indices]
        gt_labels = gt_labels[indices]

        block_ids, gt_ids = torch.where(
            block_bboxes.intersections(gt_bboxes) > 0,
        )
        block_labels = torch.zeros(
            (len(block_bboxes), num_categories),
            dtype=bool,
        )
        block_labels[block_ids, gt_labels[gt_ids]] = True

        inputs['block_labels'] = block_labels
        return inputs


# TODO: deprecate the following code


@TRANSFORMS.register_module()
class COCOTransform:

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        key: str = results['img_id']
        key = f'{key:012d}'
        results['image_id'] = key
        return results


@TRANSFORMS.register_module()
class LVISTransform:

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        key: str = results['img_path']

        prefix = 'data/lvis/'
        suffix = '.jpg'
        assert key.startswith(prefix) and key.endswith(suffix)
        key = key.removeprefix(prefix).removesuffix(suffix)

        split, key = key.split('/')
        assert split in ['train2017', 'val2017']
        split = split.removesuffix('2017')

        key = f'{split}/{key}'
        results['image_id'] = key

        return results

@TRANSFORMS.register_module()
class Objects365Transform:

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        key: str = results['img_path']
        patch: str = osp.split(osp.split(results['img_path'])[0])[1]

        assert 'train' in key or 'val' in key
        split = 'train' if 'train' in key else 'val'

        image_name = osp.basename(key)
        version = image_name.split('_')[1]
        key = image_name.removesuffix('.jpg')
        
        key = f'{split}/{version}_{patch}_{key}'
        results['image_id'] = key
        return results

class MMLoadOAKEMixin(LoadOAKEMixin[T]):

    def __init__(self, *args, access_layer: todd.Config, **kwargs) -> None:
        config = todd.Config(access_layer=access_layer)
        config = self.build_pre_hook(config, DPTransformRegistry, self)
        super().__init__(*args, access_layer=config.access_layer, **kwargs)


@TRANSFORMS.register_module()
class MMLoadOAKEGlobal(MMLoadOAKEMixin[torch.Tensor], LoadOAKEGlobal):
    pass


@TRANSFORMS.register_module()
class MMLoadOAKEBlock(MMLoadOAKEMixin[BlockOutput], LoadOAKEBlock):
    pass


@TRANSFORMS.register_module()
class MMLoadOAKEObject(MMLoadOAKEMixin[ObjectOutput], LoadOAKEObject):
    pass


@TRANSFORMS.register_module()
class MMAssignOAKEBlockLabels(AssignOAKEBlockLabels):

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        gt_bboxes = cast(BaseBoxes, results['gt_bboxes']).tensor
        inputs = dict(
            gt_bboxes=od.FlattenBBoxesXYXY(gt_bboxes),
            gt_labels=results['gt_bboxes_labels'],
            block_bboxes=results['block_bboxes'],
        )
        inputs = super().__call__(inputs)
        results['block_labels'] = inputs['block_labels']
        return results


class BBoxesFlag(enum.IntEnum):
    BLOCK = 2
    OBJECT = 3


@TRANSFORMS.register_module()
class AppendBBoxes:

    def _append_bboxes(
        self,
        gt_bboxes: torch.Tensor,
        gt_flags: torch.Tensor,
        bboxes: od.FlattenBBoxesXYXY,
        flag: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gt_bboxes = torch.cat([gt_bboxes, bboxes.to_tensor().float()])
        gt_flags = torch.cat([gt_flags,
                              torch.full((bboxes.shape[0], ), flag)])
        return gt_bboxes, gt_flags

    def __call__(self, results: dict[str, Any]) -> dict[str, Any]:
        gt_bboxes = cast(BaseBoxes, results['gt_bboxes']).tensor
        gt_flags = torch.from_numpy(results['gt_ignore_flags'])

        block_bboxes = results.pop('block_bboxes')
        object_bboxes = results.pop('object_bboxes')

        gt_bboxes, gt_flags = self._append_bboxes(
            gt_bboxes,
            gt_flags,
            block_bboxes,
            BBoxesFlag.BLOCK,
        )
        gt_bboxes, gt_flags = self._append_bboxes(
            gt_bboxes,
            gt_flags,
            object_bboxes,
            BBoxesFlag.OBJECT,
        )

        cast(BaseBoxes, results['gt_bboxes']).tensor = gt_bboxes
        results['gt_ignore_flags'] = gt_flags.numpy()
        return results


@TRANSFORMS.register_module()
class PackTrainInputs(PackDetInputs):
    OAKE_KEYS = [
        'clip_global',
        'clip_blocks',
        'clip_objects',
        'block_labels',
    ]

    def transform(self, results: dict) -> dict:
        flags = results['gt_ignore_flags']
        bboxes = results['gt_bboxes']

        block_indices = flags == BBoxesFlag.BLOCK
        object_indices = flags == BBoxesFlag.OBJECT
        indices = ~(block_indices | object_indices)

        results.update(
            gt_bboxes=bboxes[indices],
            gt_ignore_flags=flags[indices],
        )

        packed_results = super().transform(results)
        packed_results.update(
            ((k, results[k]) for k in self.OAKE_KEYS),
            block_bboxes=bboxes[block_indices],
            object_bboxes=bboxes[object_indices],
        )
        return packed_results


@TRANSFORMS.register_module()
class PackValInputs(PackDetInputs):

    def __init__(self) -> None:
        super().__init__(
            meta_keys=(
                'img_id',
                'img_path',
                'ori_shape',
                'img_shape',
                'scale_factor',
            ),
        )

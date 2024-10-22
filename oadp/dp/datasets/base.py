__all__ = [
    'BaseMixin',
    'CocoDataset',
    'LVISV1Dataset',
    'Objects365Dataset',
    'V3DetDataset',
    'DetDataPreprocessor',
]

import os
import os.path as osp
from typing import Any, Mapping

import todd
from mmdet.datasets import BaseDetDataset
from mmdet.datasets import CocoDataset as CocoDataset_
from mmdet.datasets import LVISV1Dataset as LVISV1Dataset_
from mmdet.datasets import Objects365V2Dataset as Objects365V2Dataset_
from mmdet.datasets import V3DetDataset as V3DetDataset_
from mmdet.models.data_preprocessors import (
    DetDataPreprocessor as DetDataPreprocessor_,
)
from mmdet.registry import DATASETS, MODELS
from todd.loggers import logger

from oadp.utils import Globals


class BaseMixin(BaseDetDataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            metainfo=dict(
                classes=Globals.categories.all_,
                base_classes=Globals.categories.bases,
                novel_classes=Globals.categories.novels,
            ),
            **kwargs,
        )

    @property
    def should_filter_oake(self) -> bool:
        if self.filter_cfg is None:
            return False
        return self.filter_cfg.get('filter_oake', False)

    def __len__(self) -> int:
        if todd.Store.DRY_RUN:
            return 8
        return super().__len__()

    def load_data_list(self, *args, **kwargs) -> list[dict[str, Any]]:
        data = super().load_data_list(*args, **kwargs)
        if todd.Store.DRY_RUN:
            data = data[:len(self)]
        return data


@DATASETS.register_module(force=True)
class CocoDataset(BaseMixin, CocoDataset_):

    def filter_data(self) -> list[dict[str, Any]]:
        data = super().filter_data()
        if not self.should_filter_oake:
            return data

        valid_keys = {
            k.removesuffix('.pth')
            for k in
            os.listdir('work_dirs/oake/coco/clip_objects_cuda_train/output')
            if k.endswith('.pth')
        }

        keys = {f'{datum["img_id"]:012d}': i for i, datum in enumerate(data)}

        invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
        logger.info(
            "Filtered %d invalid indices: %s",
            len(invalid_indices),
            ', '.join(data[i]['img_id'] for i in invalid_indices),
        )

        valid_data = [
            datum for i, datum in enumerate(data) if i not in invalid_indices
        ]

        return valid_data


@DATASETS.register_module(force=True)
class LVISV1Dataset(BaseMixin, LVISV1Dataset_):

    def filter_data(self) -> list[dict[str, Any]]:
        data = super().filter_data()
        if not self.should_filter_oake:
            return data

        valid_keys = {
            k.removesuffix('.pth')
            for k in
            os.listdir('work_dirs/oake/lvis/clip_objects_cuda_train/output')
            if k.endswith('.pth')
        }

        keys: dict[str, int] = dict()
        for i, datum in enumerate(data):
            key: str = datum['img_path']
            key = key.removeprefix('data/lvis/')
            key = key.removesuffix('.jpg')
            key = key.replace('/', '_', 1)
            keys[key] = i

        invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
        logger.info(
            "Filtered %d invalid indices: %s",
            len(invalid_indices),
            ', '.join(data[i]['img_path'] for i in invalid_indices),
        )

        valid_data = [
            datum for i, datum in enumerate(data) if i not in invalid_indices
        ]

        return valid_data


@DATASETS.register_module(force=True)
class Objects365Dataset(BaseMixin, Objects365V2Dataset_):

    def filter_data(self) -> list[dict[str, Any]]:
        data = super().filter_data()
        if not self.should_filter_oake:
            return data

        valid_keys = {
            k.removesuffix('.pth')
            for k in
            os.listdir('work_dirs/oake/objects365/clip_objects_cuda_train/output')
            if k.endswith('.pth')
        }

        keys: dict[str, int] = dict()
        for i, datum in enumerate(data):
            key: str = datum['img_path']
            patch: str = osp.split(osp.split(datum['img_path'])[0])[1]
            image_name = osp.basename(key)
            key = image_name.removesuffix('.jpg')
            key = f'{patch}_{key}'
            keys[key] = i

        invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
        logger.info(
            "Filtered %d invalid indices: %s",
            len(invalid_indices),
            ', '.join(data[i]['img_path'] for i in invalid_indices),
        )

        valid_data = [
            datum for i, datum in enumerate(data) if i not in invalid_indices
        ]

        return valid_data

@DATASETS.register_module(force=True)
class V3DetDataset(BaseMixin, V3DetDataset_):

    def filter_data(self) -> list[dict[str, Any]]:
        data = super().filter_data()
        if not self.should_filter_oake:
            return data

        valid_keys = {
            k.removesuffix('.pth')
            for k in
            os.listdir('work_dirs/oake/v3det/clip_objects_cuda/output')
            if k.endswith('.pth')
        }

        keys: dict[str, int] = dict()
        for i, datum in enumerate(data):
            key: str = datum['img_path']
            prefix = 'data/v3det/'
            suffix = '.jpg'
            key = key.removeprefix(prefix).removesuffix(suffix).replace('/', '_')
            keys[key] = i

        invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
        logger.info(
            "Filtered %d invalid indices: %s",
            len(invalid_indices),
            ', '.join(data[i]['img_path'] for i in invalid_indices),
        )

        valid_data = [
            datum for i, datum in enumerate(data) if i not in invalid_indices
        ]

        return valid_data

@MODELS.register_module(force=True)
class DetDataPreprocessor(DetDataPreprocessor_):

    def forward(
        self,
        data: Mapping[Any, Any],
        training: bool = False,
    ) -> dict[Any, Any]:
        return data | super().forward(data, training)

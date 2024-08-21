__all__ = [
    'BaseMixin',
    'CocoDataset',
    'LVISV1Dataset',
    'DetDataPreprocessor',
]

import os
from typing import Any

import todd
from mmdet.datasets import BaseDetDataset, CocoDataset as CocoDataset_, LVISV1Dataset as LVISV1Dataset_
from mmdet.models.data_preprocessors import DetDataPreprocessor as DetDataPreprocessor_
from mmdet.registry import DATASETS, MODELS
from todd.loggers import master_logger


class BaseMixin(BaseDetDataset):

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
            os.listdir('work_dirs/oake/coco_objects_cuda_train/output')
            if k.endswith('.pth')
        }

        keys = {f'{datum["img_id"]:012d}': i for i, datum in enumerate(data)}

        invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
        master_logger.info(
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
    pass

    # def filter_data(self) -> list[dict[str, Any]]:
    #     data = super().filter_data()
    #     if not self.should_filter_oake:
    #         return data

    #     valid_keys = {
    #         k.removesuffix('.pth')
    #         for k in
    #         os.listdir('work_dirs/oake/lvis_objects_cuda_train/output')
    #         if k.endswith('.pth')
    #     }

    #     keys: dict[str, int] = dict()
    #     for i, datum in enumerate(data):
    #         key: str = datum['img_path']
    #         key = key.removeprefix('data/lvis_v1/')
    #         key = key.removesuffix('.jpg')
    #         key = key.replace('/', '_', 1)
    #         keys[key] = i

    #     invalid_indices = {keys[key] for key in keys.keys() - valid_keys}
    #     master_logger.info(
    #         "Filtered %d invalid indices: %s",
    #         len(invalid_indices),
    #         ', '.join(data[i]['img_path'] for i in invalid_indices),
    #     )

    #     valid_data = [
    #         datum for i, datum in enumerate(data) if i not in invalid_indices
    #     ]

    #     return valid_data


@MODELS.register_module(force=True)
class DetDataPreprocessor(DetDataPreprocessor_):

    def forward(self, data: dict, training: bool = False) -> dict:
        pack_data = super().forward(data, training)
        for key, value in data.items():
            if key not in ['inputs', 'data_samples']:
                pack_data[key] = value
        return pack_data

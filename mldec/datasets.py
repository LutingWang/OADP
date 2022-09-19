from collections import namedtuple
import pathlib
from typing import Any, Dict, List, Tuple, Optional

import torchvision
import torch
import torch.utils.data.dataloader

import todd

from .debug import debug

Batch = namedtuple(
    'Batch',
    ['images', 'image_labels', 'patches', 'patch_labels', 'num_patches'],
)


class CocoClassification(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        *args,
        patches_root: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patches_root = pathlib.Path(patches_root)
        self._cat2label = {cat: i for i, cat in enumerate(self.coco.cats)}

    @property
    def classnames(self) -> List[str]:
        return [cat['name'] for cat in self.coco.cats.values()]

    @property
    def num_classes(self) -> int:
        return len(self.coco.cats)

    def _load_patch(self, id_: int) -> Dict[str, torch.Tensor]:
        return torch.load(self._patches_root / f'{id_:012d}.pth', 'cpu')

    def _load_patch_features(self, patch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return patch['patches'].float()

    def _load_patch_bboxes(self, patch: Dict[str, torch.Tensor]) -> todd.base.BBoxesXYWH:
        return todd.base.BBoxesXYWH(patch['bboxes'])

    def _load_bboxes(self, target: List[Any]) -> todd.base.BBoxesXYWH:
        return todd.base.BBoxesXYWH([anno['bbox'] for anno in target])

    def _load_bbox_labels(self, target: List[Any]) -> torch.Tensor:
        return torch.tensor([self._cat2label[anno['category_id']] for anno in target], dtype=torch.long)

    def _load_image_labels(self, bbox_labels: torch.Tensor) -> torch.Tensor:
        image_labels = torch.zeros(self.num_classes, dtype=torch.bool)
        image_labels[bbox_labels] = True
        return image_labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        id_ = self.ids[index]
        target = self._load_target(id_)
        bboxes = self._load_bboxes(target)
        bbox_labels = self._load_bbox_labels(target)
        image_labels = self._load_image_labels(bbox_labels)

        patch = self._load_patch(id_)
        patch_features = self._load_patch_features(patch)
        image_feature = patch_features[0]

        patch_bboxes = self._load_patch_bboxes(patch)
        patch_labels = torch.zeros((len(patch_bboxes), self.num_classes), dtype=torch.bool)
        patch_id, bbox_id = torch.where(patch_bboxes.intersections(bboxes) > 0)
        patch_labels[patch_id, bbox_labels[bbox_id]] = True

        return image_feature, image_labels, patch_features, patch_labels

    @staticmethod
    def collate(
        batch: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ],
    ) -> Batch:
        image_feature_list, image_labels_list, patch_features_list, patch_labels_list = zip(*batch)
        image_features: torch.Tensor = torch.utils.data.dataloader.default_collate(image_feature_list)
        image_labels: torch.Tensor = torch.utils.data.dataloader.default_collate(image_labels_list)
        patch_features = torch.cat(patch_features_list)
        patch_labels = torch.cat(patch_labels_list)
        num_patches: torch.Tensor = torch.utils.data.dataloader.default_collate([p.shape[0] for p in patch_features_list])
        if debug.CPU:
            image_features = image_features.float()
            patch_features = patch_features.float()
        return Batch(image_features, image_labels, patch_features, patch_labels, num_patches)

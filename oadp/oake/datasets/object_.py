__all__ = [
    'ObjectDataset',
    'LVISObjectDataset',
]

import os
import pickle  # nosec B403
from typing import NamedTuple
from typing import TypeVar

from PIL import Image
import todd
import todd.tasks.object_detection as od
import torch.distributed
import torch.utils.data.distributed
from todd.datasets import LVISDataset
from torch import nn

from oadp.expanded_clip import ExpandTransform
from .base import BaseDataset

from ..registries import OAKEDatasetRegistry


class T(NamedTuple):
    key: str
    proposals: torch.Tensor
    objectness: torch.Tensor
    crops: torch.Tensor
    masks: torch.Tensor


@OAKEDatasetRegistry.register_()
class ObjectDataset(BaseDataset[T]):

    def __init__(
        self,
        *args,
        proposal_file: str,
        proposal_sorted: bool,
        expand_transform: ExpandTransform,
        **kwargs,
    ) -> None:
        """Initialize.

        Args:
            proposal_file: proposal file.
            proposal_sorted: if ``True``, the first proposal corresponds to the
                image with the smallest id. Otherwise, the first image in the
                annotations file.
        """
        super().__init__(*args, **kwargs)
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)  # nosec B301
        ids = self._coco.getImgIds()
        if proposal_sorted:
            ids = sorted(ids)
        self._proposals = {
            f'{id_:012d}': torch.tensor(proposal, dtype=torch.float32)
            for id_, proposal in zip(ids, proposals)
        }

        self._expand_transform = expand_transform

    def _preprocess(self, key: str, image: Image.Image) -> T:
        proposals, objectness = self._proposals[key].split((4, 1), dim=-1)
        proposals_ = od.FlattenBBoxesXYXY(proposals)
        indices = proposals_.indices(min_wh=(4, 4))
        if todd.Store.DRY_RUN:
            indices[5:] = False
        proposals_ = proposals_[indices]
        objectness = objectness[indices]

        tensor, masks = self._expand_transform(image, proposals_)

        return T(
            key,
            proposals_.to_tensor(),
            objectness,
            tensor,
            masks,
        )


@OAKEDatasetRegistry.register_()
class LVISObjectDataset(LVISDataset, ObjectDataset):
    pass

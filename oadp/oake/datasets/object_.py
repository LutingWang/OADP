import enum
import os
import pickle  # nosec B403
from typing import NamedTuple

import einops
import PIL.Image
import todd
import todd.tasks.object_detection as od
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed

from ..registries import OADPDatasetRegistry
from .base import BaseDataset


class ObjectBatch(NamedTuple):
    id_: int
    objects: torch.Tensor
    bboxes: torch.Tensor
    objectness: torch.Tensor
    masks: torch.Tensor


class ExpandMode(enum.Enum):
    RECTANGLE = enum.auto()
    LONGEST_EDGE = enum.auto()
    CONSTANT = enum.auto()
    ADAPTIVE = enum.auto()


@OADPDatasetRegistry.register_()
class ObjectDataset(BaseDataset[ObjectBatch]):

    def __init__(
        self,
        *args,
        grid: int,
        expand_mode: str = 'ADAPTIVE',
        proposal_file: str,
        proposal_sorted: bool,
        **kwargs,
    ) -> None:
        """Initialize.

        Args:
            grid: down sampled feature map size.
            proposal_file: proposal file.
            proposal_sorted: if ``True``, the first proposal corresponds to the
                image with the smallest id. Otherwise, the first image in the
                annotations file.
            expand_mode: refer to ``ExpandMode``.
        """
        super().__init__(*args, **kwargs)
        self._grid = grid
        self._expand_mode = ExpandMode[expand_mode]
        with open(proposal_file, 'rb') as f:
            proposals = pickle.load(f)  # nosec B301
        ids = (
            self._dataset.ids
            if proposal_sorted else list(self._dataset.coco.imgs.keys())
        )
        self._proposals = {
            id_: torch.tensor(proposal, dtype=torch.float32)
            for id_, proposal in zip(ids, proposals)
        }

    def _expand(
        self,
        bboxes: od.BBoxes,
        image_wh: torch.Tensor,
    ) -> od.BBoxesXYXY:
        """Get the expanded bounding boxes.

        Args:
            bboxes: original bounding boxes.
            image_wh: width and height of the image.

        Returns:
            The expanded bounding boxes.
        """
        if self._expand_mode is ExpandMode.LONGEST_EDGE:
            length, _ = torch.max(bboxes.wh, 1, True)
        elif self._expand_mode is ExpandMode.CONSTANT:
            length = torch.full((len(bboxes), 1), 224)
        elif self._expand_mode is ExpandMode.ADAPTIVE:
            scale_ratio = 8
            length = einops.rearrange(
                torch.sqrt(bboxes.area * scale_ratio),
                'n -> n 1',
            )
        else:
            assert ValueError(self._expand_mode)

        bboxes = od.BBoxesCXCYWH(
            torch.cat([bboxes.center, length, length], dim=-1)
        )
        offset = torch.zeros_like(bboxes.lt)
        offset = torch.where(bboxes.lt >= 0, offset, -bboxes.lt)
        offset = torch.where(
            bboxes.rb <= image_wh,
            offset,
            image_wh - bboxes.rb,
        )
        offset = torch.where(bboxes.wh <= image_wh, offset, torch.tensor(0.0))
        return bboxes.translate(offset).to(od.BBoxesXYXY)

    def _object(self, image: PIL.Image.Image, bbox: od.BBox) -> torch.Tensor:
        """Crop the object and perform transformations.

        Args:
            image: original image.
            bbox: square object bounding box in `xyxy` format.

        Returns:
            Transformed image.
        """
        object_ = image.crop(bbox)
        return self._transforms(object_)

    def _mask(self, foreground: od.BBox, object_: od.BBox) -> torch.Tensor:
        r"""Crop the mask.

        Args:
            foreground: foreground bounding box in `xyxy` format.
            bbox: object bounding box in `xyxy` format with type `int`.

        Returns:
            :math:`1 \times 1 \times 1 \times h \times w` masks, where
            foreground regions are masked with 0 and background regions are 1.
        """
        x = torch.arange(object_[2] - object_[0])
        w_mask = (foreground[0] <= x) & (x <= foreground[2])
        w_mask = einops.rearrange(w_mask, 'w -> 1 w')
        y = torch.arange(object_[3] - object_[1])
        h_mask = (foreground[1] <= y) & (y <= foreground[3])
        h_mask = einops.rearrange(h_mask, 'h -> h 1')

        # 0 for the object and 1 for others
        mask = ~(w_mask & h_mask)
        mask = einops.rearrange(mask, 'h w -> 1 1 h w')
        mask = F.interpolate(
            mask.float(),
            size=(self._grid, self._grid),
            mode='nearest',
        )
        return mask

    def _preprocess(
        self,
        id_: int,
        image: PIL.Image.Image,
    ) -> ObjectBatch:
        proposals, objectness = self._proposals[id_].split((4, 1), dim=-1)
        proposals_ = od.BBoxesXYXY(proposals)
        indices = proposals_.indices(min_wh=(4, 4))
        if todd.Store.DRY_RUN:
            indices[5:] = False
        proposals_ = proposals_[indices]
        objectness = objectness[indices]

        bboxes = self._expand(proposals_, torch.tensor(image.size))
        foregrounds = proposals_.translate(-bboxes.lt).to(od.BBoxesXYXY)

        objects = []
        masks = []
        for foreground, bbox in zip(foregrounds, bboxes):
            objects.append(self._object(image, bbox))
            masks.append(self._mask(foreground, bbox))

        return ObjectBatch(
            id_,
            torch.stack(objects),
            proposals_.to_tensor(),
            objectness,
            torch.cat(masks),
        )


@OADPDatasetRegistry.register_()
class LVISObjectDataset(ObjectDataset):

    def _load_image(self, id_: int) -> PIL.Image.Image:
        info = self._dataset.coco.loadImgs([id_])[0]
        path = info['coco_url'].replace('http://images.cocodataset.org/', '')
        path = os.path.join(self._access_layer.data_root, path)
        return PIL.Image.open(path).convert("RGB")

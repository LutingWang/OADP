import enum
import math
import os
import pathlib
import pickle
from typing import NamedTuple, cast

import clip
import clip.model
import einops
import PIL.Image
import todd
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

from .base import BaseDataset, BaseValidator


class Batch(NamedTuple):
    output: pathlib.Path
    objects: torch.Tensor
    bboxes: torch.Tensor
    objectness: torch.Tensor
    masks: torch.Tensor


class ExpandMode(enum.Enum):
    RECTANGLE = enum.auto()
    LONGEST_EDGE = enum.auto()
    CONSTANT = enum.auto()
    ADAPTIVE = enum.auto()


class DatasetRegistry(todd.Registry):
    pass


@DatasetRegistry.register()
class COCODataset(BaseDataset[Batch]):

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
            proposals = pickle.load(f)
        ids = self.ids if proposal_sorted else list(self.coco.imgs.keys())
        self._proposals = {
            id_: torch.tensor(proposal, dtype=torch.float32)
            for id_, proposal in zip(ids, proposals)
        }

    def _expand(
        self,
        bboxes: todd.BBoxes,
        image_wh: torch.Tensor,
    ) -> todd.BBoxesXYXY:
        """Get the expanded bounding boxes.

        Args:
            bboxes: original bounding boxes.
            image_wh: width and height of the image.

        Returns:
            The expanded bounding boxes.
        """
        if self._expand_mode is ExpandMode.LONGEST_EDGE:
            length = torch.max(bboxes.wh, 1, True)
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

        bboxes = todd.BBoxesCXCYWH(
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
        return bboxes.translate(offset).to(todd.BBoxesXYXY)

    def _object(self, image: PIL.Image.Image, bbox: todd.BBox) -> torch.Tensor:
        """Crop the object and perform transformations.

        Args:
            image: original image.
            bbox: square object bounding box in `xyxy` format.

        Returns:
            Transformed image.
        """
        object = image.crop(bbox)
        return self.transforms.transform(object)

    def _mask(self, foreground: todd.BBox, object: todd.BBox) -> torch.Tensor:
        """_summary_

        Args:
            foreground: foreground bounding box in `xyxy` format.
            bbox: object bounding box in `xyxy` format with type `int`.

        Returns:
            :math:`1 \\times 1 \\times 1 \\times h \\times w` masks, where
            foreground regions are masked with 0 and background regions are 1.
        """
        x = torch.arange(object[2] - object[0])
        w_mask = (foreground[0] <= x) & (x <= foreground[2])
        w_mask = einops.rearrange(w_mask, 'w -> 1 w')
        y = torch.arange(object[3] - object[1])
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
        output: pathlib.Path,
        image: PIL.Image.Image,
    ) -> Batch:
        proposals, objectness = self._proposals[id_].split((4, 1), dim=-1)
        proposals_ = todd.BBoxesXYXY(proposals)
        indices = proposals_.indices(min_wh=(4, 4))
        if todd.Store.DRY_RUN:
            indices[5:] = False
        proposals_ = proposals_[indices]
        objectness = objectness[indices]

        bboxes = self._expand(proposals_, torch.tensor(image.size))
        foregrounds = proposals_.translate(-bboxes.lt).to(todd.BBoxesXYXY)

        objects = []
        masks = []
        for foreground, bbox in zip(foregrounds, bboxes):
            objects.append(self._object(image, bbox))
            masks.append(self._mask(foreground, bbox))

        return Batch(
            output,
            torch.stack(objects),
            proposals_.to_tensor(),
            objectness,
            torch.cat(masks),
        )


@DatasetRegistry.register()
class LVISDataset(COCODataset):

    def _load_image(self, id: int) -> PIL.Image.Image:
        info = self.coco.loadImgs([id])[0]
        path = info['coco_url'].replace('http://images.cocodataset.org/', '')
        return PIL.Image.open(os.path.join(self.root, path)).convert("RGB")


class Hooks:

    def __init__(self) -> None:
        self._y: torch.Tensor | None = None
        self._attn_mask: torch.Tensor | None = None

    def visual_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor]:
        attn_mask = einops.rearrange(inputs[-1], 'b 1 h w -> b (h w)')
        zeros = attn_mask.new_zeros(attn_mask.shape[0], 1)
        attn_mask = torch.cat([attn_mask, zeros], dim=-1)
        attn_mask *= -100
        self._attn_mask = attn_mask
        return inputs[:-1]

    def transformer_forward_pre(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
    ) -> None:
        x, = inputs
        self._y = x[[0]]

    def residual_attention_block_forward_pre(
        self,
        module: clip.model.ResidualAttentionBlock,
        inputs: tuple[torch.Tensor],
    ) -> None:
        assert self._y is not None
        x, = inputs
        y = self._y

        attn_mask = einops.repeat(
            self._attn_mask,
            'b v -> (b h) 1 v',
            h=module.attn.num_heads,
        )
        x = module.ln_1(torch.cat([x[1:], y]))
        y = y + module.attn(
            x[[-1]],
            x,
            x,
            need_weights=False,
            attn_mask=attn_mask,
        )[0]
        y = y + module.mlp(module.ln_2(y))
        self._y = y

    def transformer_forward(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert self._y is not None
        y = self._y
        self._y = None
        return y

    def visual_forward(
        self,
        module: clip.model.Transformer,
        inputs: tuple[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        self._attn_mask = None


class Validator(BaseValidator[Batch]):

    def __init__(self, *args, mini_batch_size: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mini_batch_size = mini_batch_size

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        config.dataset = DatasetRegistry.build(
            config.dataset,
            default_config=dict(grid=self._model.visual.grid),
        )
        return super()._build_dataloader(config)

    @classmethod
    def _build_model(
        cls,
        upsample: int = 2,
    ) -> tuple[clip.model.CLIP, transforms.Compose]:
        model, preprocess = clip.load_default(False)

        visual = model.visual
        positional_embedding = visual.interpolate_positional_embedding(
            (visual.grid * 2, ) * 2
        )
        visual.positional_embedding = nn.Parameter(positional_embedding)
        visual.grid *= upsample

        conv1 = visual.conv1
        conv1.stride = tuple(s // upsample for s in conv1.stride)
        conv1.padding = ((visual.patch_size - 1) // 2, ) * 2

        hooks = Hooks()
        visual.register_forward_pre_hook(hooks.visual_forward_pre)
        transformer = visual.transformer
        transformer.register_forward_pre_hook(hooks.transformer_forward_pre)
        transformer.register_forward_hook(hooks.transformer_forward)
        for resblock in transformer.resblocks:
            resblock = cast(clip.model.ResidualAttentionBlock, resblock)
            resblock.register_forward_pre_hook(
                hooks.residual_attention_block_forward_pre
            )

        return model, preprocess

    def _run_iter(self, batch: Batch, memo: todd.utils.Memo) -> torch.Tensor:
        objects = batch.objects
        masks = batch.masks
        if todd.Store.CUDA:
            objects = objects.cuda()
            masks = masks.cuda()
        embeddings = []
        for i in range(math.ceil(objects.shape[0] / self._mini_batch_size)):
            indices = slice(
                i * self._mini_batch_size,
                (i + 1) * self._mini_batch_size,
            )
            o = objects[indices].type(self._model.dtype)
            m = masks[indices].type(self._model.dtype)
            embedding = self._model.visual(o, m)
            embedding = F.normalize(embedding)
            embeddings.append(embedding)
        memo['result'] = dict(
            embeddings=torch.cat(embeddings).half(),
            bboxes=batch.bboxes.half(),
            objectness=batch.objectness.half()
        )
        return super()._run_iter(batch, memo)


if __name__ == '__main__':
    Validator.main()

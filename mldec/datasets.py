import random
from typing import Any, List, Tuple

import numpy as np
from torchvision import datasets as datasets
import torch
import torch.utils.data
import torch.utils.data.dataloader
import torch.nn.functional as F
from PIL import Image, ImageDraw

import todd


class CocoClassification(datasets.coco.CocoDetection):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._cat2label = {cat: i for i, cat in enumerate(self.coco.cats.keys())}

    @property
    def classnames(self) -> List[str]:
        return [cat['name'] for cat in self.coco.cats.values()]

    def _load_target(self, *args, **kwargs) -> torch.Tensor:
        target = super()._load_target(*args, **kwargs)
        bbox_labels = [self._cat2label[anno['category_id']] for anno in target]
        image_labels = torch.zeros(len(self._cat2label), dtype=torch.bool)
        image_labels[bbox_labels] = True
        return image_labels


class PatchedCocoClassification(CocoClassification):
    def __init__(
        self,
        *args,
        patch_size: int = 224,
        max_stride: int = 224,
        rescale: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._patch_size = patch_size
        self._max_stride = max_stride
        self._rescale = rescale

    def _cut_patches(self, length: int) -> List[int]:
        assert length >= self._patch_size
        if length == self._patch_size:
            return [0]
        n = (length - self._patch_size - 1) // self._max_stride + 1
        stride = (length - self._patch_size) // n
        mod = (length - self._patch_size) % n

        result = [0]
        for i in range(n):
            result.append(result[-1] + stride + (i < mod))
        return result

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        id_ = self.ids[index]
        image = self._load_image(id_)
        target = self._load_target(id_)

        patches = [image]
        while image.width >= self._patch_size and image.height >= self._patch_size:
            for x in self._cut_patches(image.width):
                for y in self._cut_patches(image.height):
                    patch_box = (x, y, x + self._patch_size, y + self._patch_size)
                    patch = image.crop(patch_box)
                    patches.append(patch)
            rescaled_size = (int(image.width / self._rescale), int(image.height / self._rescale))
            image = image.resize(rescaled_size)
        patches = list(map(self.transforms.transform, patches))

        return patches, target

    @staticmethod
    def collate(batch: List[Tuple[List[torch.Tensor], torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches_list, targets = zip(*batch)
        patches = sum(patches_list, [])
        num_patches = list(map(len, patches_list))
        return tuple(map(torch.utils.data.dataloader.default_collate, (patches, targets, num_patches)))


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class Convert:

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB")

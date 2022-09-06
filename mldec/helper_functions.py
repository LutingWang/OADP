import random
from typing import Any, List, Tuple

import numpy as np
from torchvision import datasets as datasets
import torch
import torch.utils.data
import torch.utils.data.dataloader
from PIL import ImageDraw

import todd


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.sum = None
        self.cnt = None
        self.avg = None
        self.ema = None
        self.initialized = False

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def initialize(self, val, n):
        self.val = val
        self.sum = val * n
        self.cnt = n
        self.avg = val
        self.ema = val
        self.initialized = True

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        self.ema = self.ema * 0.99 + self.val * 0.01


class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._max_stride = 224
        self._patch_size = 224
        self._rescale = 1.5

        self._cat2label = dict()
        for cat in self.coco.cats.keys():
            self._cat2label[cat] = len(self._cat2label)

    def _cut(self, length: int) -> List[int]:
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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        empty = len(target) == 0
        if empty:
            target = [dict(bbox=[0, 0, 0, 0], category_id=1)]

        bboxes = torch.tensor([anno['bbox'] for anno in target])
        bbox_labels = torch.tensor([self._cat2label[anno['category_id']] for anno in target], dtype=torch.long)

        patches = [self.transforms.transform(image)]
        patch_labels = [bbox_labels]
        while image.width >= self._patch_size and image.height >= self._patch_size:
            bboxes_xywh = todd.base.BBoxesXYWH(bboxes)
            for x in self._cut(image.width):
                for y in self._cut(image.height):
                    patch = image.crop((x, y, x + self._patch_size, y + self._patch_size))
                    patches.append(self.transforms.transform(patch))
                    patch_bbox_xywh = todd.base.BBoxesXYWH(torch.tensor([[x, y, self._patch_size, self._patch_size]]))
                    patch_labels.append(bbox_labels[bboxes_xywh.intersections(patch_bbox_xywh).squeeze(-1) > 0])
            rescaled_size = (int(image.width / self._rescale), int(image.height / self._rescale))
            image = image.resize(rescaled_size)
            bboxes = bboxes / self._rescale

        for i, label in enumerate(patch_labels):
            label_ = torch.zeros(80, dtype=torch.bool)
            if not empty:
                label_[label] = 1
            patch_labels[i] = label_

        return patches, patch_labels

    @staticmethod
    def collate(batch: List[Tuple[List[torch.Tensor], List[torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches, patch_labels = zip(*batch)
        num_patches = list(map(len, patches))
        patches = sum(patches, [])
        patch_labels = sum(patch_labels, [])
        return tuple(map(torch.utils.data.dataloader.default_collate, (patches, patch_labels, num_patches)))


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

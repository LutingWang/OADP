__all__ = [
    'MultilabelTopKRecall',
    'NormalizedLinear',
]
import random
from typing import Any

import cv2
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.datasets.coco import CocoDataset

import todd


class MultilabelTopKRecall(todd.Module):

    def __init__(self, *args, k: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._k = k

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the multilabel topk recall.

        Args:
            logits: :math:`bs \\times K`, float.
            targets: :math:`bs \\times K`, bool.

        Returns:
            One element tensor representing the recall.
        """
        _, indices = logits.topk(self._k)
        preds = torch.zeros_like(targets).scatter(1, indices, 1)
        # labels showing up at least once
        labels, = torch.where(targets.sum(0))
        recall = sklearn.metrics.recall_score(
            targets.cpu().numpy(),
            preds.cpu().numpy(),
            labels=labels.cpu().numpy(),
            average='macro',
            zero_division=0,
        )
        return logits.new_tensor(recall * 100)


class NormalizedLinear(nn.Linear):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        x = super().forward(*args, **kwargs)
        return F.normalize(x)


def draw_label_type(img: Any, bbox: list, label: str, color: tuple) -> Any:
    font = cv2.FONT_HERSHEY_SIMPLEX
    labelSize = cv2.getTextSize(label + '0', font, 0.5, 2)[0]
    img = cv2.rectangle(
        img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2
    )
    if bbox[1] - labelSize[1] - 3 < 0:
        x1, y1, x2, y2 = bbox[0], bbox[1] + 2, bbox[0] + \
            labelSize[0], bbox[1] + labelSize[1]
    else:
        x1, y1, x2, y2 = bbox[0], bbox[1] - labelSize[1] - \
            3, bbox[0] + labelSize[0], bbox[1]

    img = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=-1)
    img = cv2.putText(img, label, (x1, y2), font, 0.5, (0, 0, 0), thickness=1)
    return img


def plot_single_img(
    image_path: str, bbox_result: list[Any], threshold: float,
    output_path: str, categories: list | tuple
) -> None:
    image = cv2.imread(image_path)
    PALETTE = CocoDataset.PALETTE
    for idx, bboxes in enumerate(bbox_result):
        color = random.choice(PALETTE)
        for bbox in bboxes:
            x1, y1, x2, y2, score = list(bbox)
            if score >= threshold:
                image = draw_label_type(
                    image, [int(x1), int(y1),
                            int(x2), int(y2)],
                    '{}|{:.2}'.format(categories[idx], score), color
                )
    cv2.imwrite(output_path, image)

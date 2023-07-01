import os
from typing import NamedTuple

import todd
import torch
from torchvision.datasets import CocoDetection

from .classifiers import ClassifierRegistry


class Batch(NamedTuple):
    proposal_embeddings: torch.Tensor
    proposal_objectness: torch.Tensor
    proposal_bboxes: torch.Tensor
    image_ids: torch.Tensor
    class_embeddings: torch.Tensor
    scaler: torch.Tensor
    bias: torch.Tensor


class Dataset(CocoDetection):
    _classnames: list[str]
    _cat2label: dict[int, int]

    def __init__(
        self,
        root: str,
        ann_file: str,
        proposal: str,
        classifier: todd.Config,
        top_KP: int = 100,
    ) -> None:
        super().__init__(root=root, annFile=ann_file)
        self._ann_file = ann_file
        self.top_KP = top_KP
        self.proposal_root = proposal
        self.classifier = ClassifierRegistry.build(classifier)

    @property
    def classnames(self) -> tuple[str, ...]:
        return tuple(self._classnames)

    @property
    def num_classes(self) -> int:
        return len(self._classnames)

    def __len__(self) -> int:
        if todd.Store.DRY_RUN:
            return 50
        return len(self.ids)

    def _load_target(self, *args, **kwargs) -> list:
        target = super()._load_target(*args, **kwargs)
        return [
            anno for anno in target if anno['category_id'] in self._cat2label
        ]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id = torch.tensor(self.ids[index])
        proposal_pth = f'{image_id.item():012d}.pth'
        data_path = os.path.join(self.proposal_root, proposal_pth)
        data_ckpt = torch.load(data_path, 'cpu')

        proposal_embeddings = torch.Tensor(data_ckpt['embeddings'])
        proposal_objectness = torch.Tensor(data_ckpt['objectness'])
        proposal_bboxes = torch.Tensor(data_ckpt['bboxes'])
        inds = torch.arange(self.top_KP)

        return (
            proposal_embeddings[inds], proposal_objectness[inds],
            proposal_bboxes[inds], image_id
        )

    def collate(
        self, batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor]]
    ) -> Batch:
        proposal_embeddings, proposal_objectness, \
            proposal_bboxes, image_ids = map(
                torch.stack, zip(*batch)
            )
        return Batch(
            proposal_embeddings, proposal_objectness, proposal_bboxes,
            image_ids, *self.classifier.infos
        )

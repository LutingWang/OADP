import os
from collections import namedtuple
from typing import Any, Dict, List, Optional, Tuple

import todd
import torch
import torchvision
from lvis.lvis import LVIS

from ..base import globals_ as cats

Batch = namedtuple(
    'Batch',
    [
        'proposal_embeddings', 'proposal_objectness', 'proposal_bboxes',
        'image_ids', 'class_embeddings', 'scaler', 'bias'
    ],
)


class PseudoLabelDataset(torchvision.datasets.CocoDetection):
    _classnames: List[str]
    _cat2label: Dict[int, int]

    def __init__(
        self,
        type: str,
        root: str,
        ann_file: str,
        pretrained: str,
        proposal: Optional[str] = None,
        coco_split: Optional[str] = None,
        lvis_ann_file: Optional[str] = None,
        lvis_split: Optional[str] = None,
        top_KP: int = 100,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )
        self._ann_file = ann_file
        self.top_KP = top_KP

        if type == "coco":
            classnames = getattr(cats, coco_split)
            self._classnames = []
            self._cat2label = dict()
            for cat in self.coco.cats.values():
                if cat['name'] in classnames:
                    self._classnames.append(cat['name'])
                    self._cat2label[cat['id']] = len(self._cat2label)

        elif type == "lvis":
            classnames = getattr(cats, lvis_split)
            self._lvis = LVIS(lvis_ann_file)
            self._classnames = []
            self._cat2label = dict()
            for cat in self._lvis.cats.values():
                if cat['name'] in classnames:
                    self._classnames.append(cat['name'])
                    self._cat2label[cat['id']] = len(self._cat2label)
        else:
            raise RuntimeError("please choose split")

        self._label2cat = dict()
        for cat_id in self._cat2label.keys():
            self._label2cat[self._cat2label[cat_id]] = cat_id

        ckpt = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = ckpt['embeddings']
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in self._classnames]
        self._class_embeddings = embeddings[inds].half()
        new_names = []
        for in_ in inds:
            new_names.append(ckpt['names'][in_])
        assert new_names == self._classnames

        if 'scaler' in ckpt.keys():
            self._scaler = ckpt['scaler'].item()
        else:
            self._scaler = 1
        if 'bias' in ckpt.keys():
            self._bias = ckpt['bias'].item()
        else:
            self._bias = 0
        self.proposal_root = proposal

    @property
    def classnames(self) -> Tuple[str]:
        return tuple(self._classnames)

    @property
    def num_classes(self) -> int:
        return len(self._classnames)

    def __len__(self) -> int:
        if todd.Store.DRY_RUN:
            return 3
        return len(self.ids)

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [
            anno for anno in target if anno['category_id'] in self._cat2label
        ]

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        image_id = torch.tensor(self.ids[index])
        proposal_pth = f'{image_id.item():012d}.pth'
        data_path = os.path.join(
            self.proposal_root, "train2017/", proposal_pth)
        if not os.path.exists(data_path):
            raise RuntimeError(f"data_path:{data_path} not exist")
        data_ckpt = torch.load(data_path, 'cpu')
        proposal_embeddings = data_ckpt['embeddings']
        proposal_objectness = data_ckpt['objectness']
        proposal_bboxes = data_ckpt['bboxes']
        inds = torch.arange(self.top_KP)

        return (proposal_embeddings[inds],
                proposal_objectness[inds],
                proposal_bboxes[inds],
                image_id)

    def collate(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Batch:
        proposal_embeddings, proposal_objectness, proposal_bboxes, image_ids = map(
            torch.stack, zip(*batch)
        )
        return Batch(
            proposal_embeddings, proposal_objectness, proposal_bboxes,
            image_ids, self._class_embeddings, self._scaler, self._bias
        )

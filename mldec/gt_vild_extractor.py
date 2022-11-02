import argparse
import enum
import math
import os
import pathlib
import pickle
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, cast

import PIL.Image
import clip
import clip.model
import einops
import todd
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from mmdet.core import bbox2result
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.utils.class_weight import compute_sample_weight
from .debug import debug
from .todd import BaseRunner, TrainerMixin
from .utils import odps_init, k8s_init
from . import datasets

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC


class Batch(NamedTuple):
    patches: torch.Tensor
    bboxes: torch.Tensor
    bboxes_labels: torch.Tensor
    class_embeddings: torch.Tensor
    


class ExpandMode(enum.IntEnum):
    VILD = enum.auto()
    CLIP = enum.auto()
    RECTANGLE = enum.auto()
    LONGEST_EDGE = enum.auto()
    CONSTANT = enum.auto()
    ADAPTIVE = enum.auto()



class CocoClassification(torchvision.datasets.CocoDetection):

    def __init__(
        self,
        root: str,
        ann_file: str,
        mode: str,
        split:Optional[str],
        pretrained:Optional[str],

    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )
        if split is None:
            self._classnames = [cat['name'] for cat in self.coco.cats.values()]
            self._cat2label = {cat: i for i, cat in enumerate(self.coco.cats)}
        else:
            classnames = getattr(datasets, split)
            self._classnames = []
            self._cat2label = dict()
            for cat in self.coco.cats.values():
                if cat['name'] in classnames:
                    self._classnames.append(cat['name'])
                    self._cat2label[cat['id']] = len(self._cat2label)

        ckpt = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = ckpt['embeddings']
        name2ind = {name: i for i, name in enumerate(ckpt['names'])}
        inds = [name2ind[name] for name in self._classnames]
        self._class_embeddings = embeddings[inds]


        self.mode = ExpandMode[mode.upper()]


        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    @property
    def embeddings_root(self) -> pathlib.Path:
        return self._embeddings_root

    def _crop(
        self,
        image: PIL.Image.Image,
        bboxes: todd.BBoxesXYXY,
    ) -> torch.Tensor:
        return torch.stack([
            self._transform(image.crop(bbox))
            for bbox in bboxes.round().to_tensor().int().tolist()
        ])

    def _enlarge(
        self,
        image: PIL.Image.Image,
        bboxes: todd.BBoxesXYXY,
        **kwargs,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: image to be cropped
            bboxes: n x 4

        Returns:
            patches: n x 3 x 224 x 224, cropped bboxes
            masks: n x 197, processed transformer masks
        """

        if self.mode == ExpandMode.LONGEST_EDGE:
            max_length = einops.rearrange(
                torch.max(bboxes.wh, 1).values,
                'n -> n 1',
            )
        elif self.mode == ExpandMode.CONSTANT:
            max_length = torch.full((len(bboxes), 1), 224)
        elif self.mode == ExpandMode.ADAPTIVE:
            scale_ratio = kwargs.get('scale_ratio', 8)
            max_length = einops.rearrange(
                torch.sqrt(bboxes.area * scale_ratio),
                'n -> n 1',
            )
        else:
            assert ValueError(self._expand_mode)

        lt = bboxes.center - max_length / 2
        rb = bboxes.center + max_length / 2

        image_size = torch.tensor(image.size)
        offset = torch.zeros(len(bboxes), 2)
        offset = torch.where(lt >= 0, offset, lt)
        offset = torch.where(rb <= image_size, offset, rb - image_size)
        offset = torch.where(max_length <= min(image_size), offset, torch.tensor(0.0))

        lt -= offset
        rb -= offset
        expanded_bboxes = torch.cat([lt, rb], -1)
        return expanded_bboxes   
    

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data = super().__getitem__(index) 
        image_id = self.ids[index]
        
        image, target = data
        # breakpoint()
        bboxes = torch.tensor([[bbox['bbox'][0],bbox['bbox'][1],bbox['bbox'][0]+bbox['bbox'][2],bbox['bbox'][1]+bbox['bbox'][3]] for bbox in target]) 
        bboxes_labels =torch.tensor([bbox['category_id'] for bbox in target])
        
        # raise RuntimeError("check bbox format")
        if bboxes.shape[0] == 0:
            return None
        proposals = todd.BBoxesXYXY(bboxes[:, :4])
        indices = proposals.indices(min_wh=(4, 4))
        proposals = proposals[indices]
        # if proposals.shape[0] == 0:
        #     return None
        bboxes_labels = bboxes_labels[indices]
        # import ipdb;ipdb.set_trace()
        bboxes_labels= torch.tensor([self._cat2label[label.item()] for label in bboxes_labels])
        if self.mode == ExpandMode.VILD:
            patches = torch.stack([
                self._crop(image, proposals),
                self._crop(image, proposals.expand(1.5)),
            ])
        elif self.mode == ExpandMode.CLIP:
            patches = torch.stack([
                self._crop(image, proposals),
                self._crop(image, proposals),
            ])
        else:
            proposals = todd.BBoxesXYXY(self._enlarge(image,proposals))
            patches = torch.stack([
                self._crop(image, proposals),
                self._crop(image, proposals),
            ])
            # raise RuntimeError("check mode format")
        # import ipdb;ipdb.set_trace()
        return patches, proposals.to_tensor(),bboxes_labels

    # @staticmethod
    # def collate(batch: List[Optional[Batch]]) -> Optional[Batch]:
    #     assert len(batch) == 1
    #     return batch[0]

    # @staticmethod
    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
        batch = [item for item in batch if item != None]
        patches, proposals,labels = zip(*batch)
        return Batch(torch.cat(patches,dim=1),torch.cat(proposals),torch.cat(labels),self._class_embeddings)


class Model(todd.reproduction.FrozenMixin, todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        self._visual = clip_model.visual

        frozen_names = ['_visual']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=frozen_names),
            eval_config=dict(names=frozen_names),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._visual(x)
        return F.normalize(x)


class Runner(BaseRunner):

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(**config.dataset)
        sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=False,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=dataset.collate,
        )
        return dataset, sampler, dataloader

    def _build_model(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        config = config.copy()
        pretrained = config.pop('pretrained')
        assert "ViT-B/32" in pretrained
        if debug.CPU:
            clip_model, _ = clip.load(pretrained, 'cpu')
        else:
            clip_model, _ = clip.load(pretrained)
        clip_model.requires_grad_(False)
        model = Model(clip_model=clip_model, **config)
        if not debug.CPU:
            model = model.cuda()
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Optional[Batch], memo: Dict[str, Any], **kwargs,
    ) -> None:
        if batch is None:
            return
        patches = batch.patches
        if debug.DRY_RUN:
            patches = patches[:5]
            masks = masks[:5]
        if not debug.CPU:
            patches = patches.half().cuda()
            bboxes_labels = batch.bboxes_labels.cuda()
            class_embeddings = batch.class_embeddings.half().cuda()
            bboxes = batch.bboxes.half().cuda()

        n = patches.shape[0]
        # import ipdb;ipdb.set_trace()
        patches = einops.rearrange(patches, 'n b c h w -> (n b) c h w')
        patches = self._model(patches)
        patches = einops.reduce(patches, '(n b) c -> b c', n=n, reduction='mean')
        
        score = patches@class_embeddings.T
        logit = score.softmax(dim=-1)
        pre_score,pre_cls = torch.max(logit,dim=1)
        # import ipdb;ipdb.set_trace()
        final_bboxes = torch.cat((bboxes,pre_score[...,None],pre_cls[...,None],bboxes_labels[...,None]),dim=-1)
        # final_bboxes = torch.cat((batch.bboxes,logits[...,None],batch.bbox_labels[...,None]),dim=-1)
        memo['results'].append(final_bboxes)

    def _after_run_iter(self, *args, i: int, batch: Optional[Batch], memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:

        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:

        results = memo['results']
        # import ipdb;ipdb.set_trace()
        # bbox_results = [bbox2result(result[...,:-2], result[...,-2], self._dataset.num_classes) for result in results]

        # self._dataset.evaluate(bbox_results)

        results = torch.cat(memo['results'])
        logits, labels = results[...,-2].detach().cpu(),results[...,-1].detach().cpu()

        macro_mAP = precision_score(labels,logits,average="macro")
        weighted_mAP = precision_score(labels,logits,average="weighted")
        self._logger.info(f"macro_mAP {macro_mAP * 100:.3f}")
        self._logger.info(f"weighted_mAP {weighted_mAP * 100:.3f}")
        sw = compute_sample_weight(class_weight='balanced',y=labels)
        cm = confusion_matrix(labels,logits, sample_weight=sw)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--k8s', action=todd.base.DictAction)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    if args.k8s is not None:
        k8s_init(args.k8s)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        if args.local_rank is not None:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
        torch.cuda.set_device(todd.base.get_local_rank())

    todd.reproduction.init_seed(args.seed)

    runner = Runner(name=args.name, config=config)

    runner.run()

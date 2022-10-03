from collections import namedtuple
from typing import Any, Dict, Iterator, List, Optional, Tuple
from enum import Enum
import pickle
import argparse
import pathlib

from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import torchvision
import torch.distributed
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image

import clip
import clip.model
import todd
from . import datasets
from .debug import debug
from .utils import all_gather, odps_init
from .todd import BaseRunner
from .spatial_vit import CLIPMaskedSpatialViT
from mmdet.core import  build_assigner
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.core.bbox.assigners import MaxIoUAssigner
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC


Batch = namedtuple(
    'Batch',
    ['images', 'masks', 'bboxes', 'bbox_labels', 'class_embeddings', 'scaler', 'bias','objectness','image_id'],
)
class Box(Enum):
    RECTANGLE = 0
    LONGEST_EDGE = 1
    CONSTANT = 2
    ADAPTIVE= 3


def getSquareBox(bboxes:torch.Tensor,image_size:List[int],n_px:int=224,box_type=Box.LONGEST_EDGE)-> Tuple[torch.Tensor, torch.Tensor] :
    """
    Args:
        bboxes: n x 4
        image_size: width height 2

    Returns:
        box_r: n x 4
        mask_pool: list 

    """
    rb = bboxes[:,2:]
    lt = bboxes[:,:2]
    wh = rb-lt
    center = (lt+rb)/2
    image_size = torch.tensor(image_size,dtype = center.dtype,device  = center.device)
    if box_type == Box.RECTANGLE:#rectangle
        box_r = bboxes
        mask_pool = torch.ones((box_r.shape[0],n_px,n_px))
    else:
        if box_type == Box.LONGEST_EDGE:    
            max_length = torch.max(wh,dim =1)[0][...,None]

        elif box_type == Box.CONSTANT:
            """
            make a 224*224 box
            the calculation is same to type 0 

            """
            max_length = n_px

        elif box_type == Box.ADAPTIVE:
            """
            adaptive

            """
            scale_ratio = 8
            max_length = torch.sqrt(wh[:,0]*wh[:,1]*scale_ratio)[...,None]
            assert torch.nan not in max_length

        pse_lt = center - max_length/2
        pse_rb = center + max_length/2
        offset_ = torch.tensor([[0,0]],dtype=center.dtype,device=center.device).repeat(wh.shape[0],1)
        offset = torch.where(pse_rb>image_size,pse_rb-image_size,offset_)
        offset = torch.where(pse_lt<0,pse_lt,offset)
        offset = torch.where(max_length > min(image_size),offset_,offset)

        new_center = center - offset
        box_r = torch.cat((new_center - max_length/2,new_center + max_length/2),axis = -1)
        #get mask
        box_mask_lt = torch.where(lt - box_r[:,:2] > 0,lt - box_r[:,:2],torch.zeros_like(lt))
        box_mask_rb = torch.where(rb - box_r[:,:2] > 0,rb - box_r[:,:2],torch.zeros_like(lt))
        box_mask = torch.cat((box_mask_lt,box_mask_rb),dim =-1)

        mask_pool = []
        for i,(length,height) in enumerate(zip(box_r.round().int()[:,2]-box_r.round().int()[:,0],box_r.round().int()[:,3]-box_r.round().int()[:,1])):
            col = torch.reshape(torch.arange(length),(1,-1)).repeat(height,1)[...,None]
            row = torch.reshape(torch.arange(height),(-1,1)).repeat(1,length)[...,None]
            index = torch.cat((row,col),dim =-1)
            mask_pool.append(torch.where((index[...,1]>=box_mask[i,0]) & (index[...,0]>=box_mask[i,1]) &(index[...,1]<=box_mask[i,2]) &(index[...,0]<=box_mask[i,3]),torch.ones((height,length)),torch.zeros((height,length))))
 
    return box_r,mask_pool


def masks_to_attn_map(masks:torch.Tensor,_target_size=14)-> torch.Tensor:
    # masks size NxHxW
    N = masks.size(0)
    assert N==1 
    # masks is 1 for the object and 0 for others, need to invert it
    masks = 1 - masks.bool().float()
    # interpolate to target size
    masks = masks.unsqueeze(1).float()
    target_size = (_target_size, _target_size)
    masks = F.interpolate(masks, size=target_size,
                        mode='nearest', align_corners=None)
    masks = masks.squeeze(1)
    attn_map = masks.view(N, -1)
    attn_map = torch.cat([attn_map, 1-torch.eye(N).to(attn_map.device)], 1)
    attn_map = attn_map.bool().half() * (-100)
    return attn_map[0]


def prep(
    image: PIL.Image.Image,
    bboxes: torch.Tensor,
    bbox_labels: torch.Tensor,
    transform,
    objectness,
    image_id,
    n_px : int = 224,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]:
    """
    Args:
        image: original image
        bboxes: n x 4
        bbox_labels: n
        n_px : 224

    Returns:
        images: n' x 3 x 224 x 224, cropped bboxes
        masks: n' x 197, processed transformer masks
        bboxes: n' x 4, filtered bboxes
        bbox_labels: n', filtered labels
    """

    if bboxes.shape[0] == 0 : 
        images_r = torch.empty([0,3,n_px,n_px])
        masks_r = torch.empty([0,197])
        bboxes_r = torch.empty([0,4])
        bbox_labels_r = torch.empty([0])
    else:
       
        assert bboxes.shape[0] == bbox_labels.shape[0]
        
        bboxes_r = bboxes
        bbox_labels_r = bbox_labels 

        images_r = []
        masks_r = []
        
        bboxes_s , masks_s = getSquareBox(bboxes_r,[image.width,image.height],n_px=n_px,box_type=Box.ADAPTIVE)
        for i,(box_s, mask_s) in enumerate(zip(bboxes_s,masks_s)):
            box_r= torch.round(box_s).int().tolist()

            image_r = image.crop(box_r)
            images_r.append(transform(image_r)) 
            
            mask_r = masks_to_attn_map(mask_s[None].bool())
            masks_r.append(mask_r)
        if len(images_r)!= 0:
            images_r = torch.stack(images_r)
            masks_r = torch.stack(masks_r)
        else:
            images_r = torch.empty([0,3,n_px,n_px])
            masks_r = torch.empty([0,197])
            bboxes_r = torch.empty([0,4])
            bbox_labels_r = torch.empty([0])
    
    return images_r,masks_r,bboxes_r,bbox_labels_r,objectness,image_id


class CocoClassification(torchvision.datasets.CocoDetection):
    _classnames: List[str]
    _cat2label: Dict[int, int]

    def __init__(
        self,
        root: str,
        ann_file: str,
        pretrained: str,
        proposal: str = None,
        split: Optional[str] = None,
        n_px: int = 224,
        assigner: dict = None
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
        self._scaler = ckpt['scaler'].item()
        self._bias = ckpt['bias'].item()
        self._n_px = n_px
        if assigner != None:
            self.assigner: MaxIoUAssigner = build_assigner(assigner)
        else:
            self.assigner = None
        if proposal != None:
            with open(proposal,'rb') as f:
                self.proposals = pickle.load(f)
        else:
            self.proposals = None
        def _convert_image_to_rgb(image):
            return image.convert("RGB")

        def _transform(n_px):
            return Compose([
                Resize(n_px, interpolation=BICUBIC),
                CenterCrop(n_px),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        self._preprocess = _transform(n_px)

    @property
    def classnames(self) -> Tuple[str]:
        return tuple(self._classnames)

    @property
    def num_classes(self) -> int:
        return len(self._classnames)

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [anno for anno in target if anno['category_id'] in self._cat2label]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]:
        image, target = super().__getitem__(index)

        image_id = torch.tensor([self.ids[index]])
        bboxes = torch.tensor([anno['bbox'] for anno in target])
        if bboxes.shape[0]!=0:
            bboxes = torch.stack((bboxes[:,0],bboxes[:,1],bboxes[:,0]+bboxes[:,2],bboxes[:,1]+bboxes[:,3]),dim =-1) 
        bbox_labels = torch.tensor([self._cat2label[anno['category_id']] for anno in target])
        if self.assigner!= None:
            anchors = torch.tensor(self.proposals[index],dtype = bboxes.dtype,device=bboxes.device)
            anchors_ = torch.stack((anchors[:,0],anchors[:,1],anchors[:,0]+anchors[:,2],anchors[:,1]+anchors[:,3]),dim =-1)

            if bboxes.shape[0]!= 0:
                assign_result = self.assigner.assign(anchors_[:,:4], bboxes, gt_labels = bbox_labels)
                return prep(image, anchors_[:,:4], assign_result.labels,self._preprocess,anchors[:,-1],image_id)
            else:
                assign_result_labels = -1*torch.ones_like(anchors_[:,0])
                anchors_ = anchors_
                return prep(image, anchors_[:,:4], assign_result_labels,self._preprocess,anchors[:,-1],image_id)
        else:
            return prep(image, bboxes, bbox_labels,self._preprocess,None,image_id)

    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
        images, masks, bboxes, bbox_labels, bbox_objectness, image_id = map(torch.cat, zip(*batch))
        return Batch(images, masks, bboxes, bbox_labels, self._class_embeddings, self._scaler, self._bias, bbox_objectness,image_id)


class Model(todd.base.Module):

    def __init__(
        self,
        *args,
        config: todd.base.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        clip_model, _ = clip.load(config.pretrained, 'cpu')
        assert "ViT-B" in config.pretrained and "32" in config.pretrained
        clip_model.requires_grad_(False)
        self._model = CLIPMaskedSpatialViT(clip_model,patch_size = 32,upsample=2)
        self.tp = 0
        self.fp = 0

    def forward(self, batch: Batch) -> torch.Tensor:
        
        images = batch.images
        masks = batch.masks
        bbox_labels = batch.bbox_labels
        assert images.shape[0] == masks.shape[0]
        assert images.shape[0] == bbox_labels.shape[0]
        image_features = []
        for i,(img,mask) in enumerate(zip(images,masks)):
            
            image_feature = self._model(img.unsqueeze(0), mask.unsqueeze(0))
            image_feature = image_feature.permute(0, 2, 1)

            image_feature = (
                image_feature / image_feature.norm(dim=1, keepdim=True)
            )
            image_features.append(image_feature)

        image_features = torch.concat(image_features)
        return image_features


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
        model = Model(config=config).requires_grad_()
        if not debug.CPU:
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        embeddings_root = pathlib.Path(self._config.embeddings_root) / 'val'
        embeddings_root.mkdir(parents=True, exist_ok=True)
        memo['embeddings_root'] = embeddings_root
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in batch])
        proposal_embedding = self._model(batch)
        # breakpoint()
        memo['results']=dict(
            proposal_bboxes=batch.bboxes.half(),
            proposal_embeddings=proposal_embedding[...,0].half(),
            proposal_labels = batch.bbox_labels.half(),
            proposal_objectness = batch.objectness.half()
        )

    def _after_run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs) -> None:
        torch.save(
            memo.pop('results'),
            memo['embeddings_root'] / f'{batch.image_id[0]:012d}.pth',
        )
        if todd.base.get_rank() == 0 and i % self._config.log_interval == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    runner = Runner(name=args.name, config=config)
    runner.run()

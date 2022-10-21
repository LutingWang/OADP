import argparse
from asyncio.log import logger
from collections import namedtuple
from logging import raiseExceptions
import pathlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, final
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed
import torch.nn as nn
import PIL.Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import clip
import clip.model
import os
import todd
import torchvision
from mmdet.core import bbox2result
from . import datasets
from mmdet.datasets.coco import CocoDataset
from .debug import debug
from .utils import odps_init
from .todd import BaseRunner
from mmdet.core import  multiclass_nms
import torch.distributed as dist
from mmcv.runner import get_dist_info
# from ..todd import BBoxesXYXY
import torch.nn.functional as F
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.utils.class_weight import compute_sample_weight
from mmdet.core.visualization import imshow_det_bboxes
import json
import nni
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from lvis.lvis import LVIS
from lvis.results import LVISResults
from lvis.eval import LVISEval
from lvis.vis import LVISVis
Batch = namedtuple(
    'Batch',
    [ 'proposal_embeddings', 'proposal_objectness','proposal_bboxes','image_ids','class_embeddings', 'scaler', 'bias','mask'],
)

def all_gather(tensors_: Tuple[torch.Tensor],shape:torch.Tensor) -> List[torch.Tensor]:
    # import ipdb;ipdb.set_trace()
    tensor = torch.cat(tensors_)
    tensors = []
    for _ in range(todd.base.get_world_size()):
        if len(tensor.shape)==2:
            tensors.append(torch.zeros([int(shape),tensor.shape[1]],device = tensor.device,dtype = tensor.dtype))
        else:
            tensors.append(torch.zeros([int(shape)],device = tensor.device,dtype = tensor.dtype))
        # print(todd.get_rank(),tensors[_].shape)
    if len(tensor.shape)==2:
        fake_tensor = torch.zeros([int(shape)-tensor.shape[0],tensor.shape[1]],device = tensor.device,dtype = tensor.dtype)
    else:
        fake_tensor = torch.zeros([int(shape)-tensor.shape[0]],device = tensor.device,dtype = tensor.dtype)
    tensor = torch.cat((tensor,fake_tensor))
    torch.distributed.all_gather(tensors, tensor)
    return tensors

def all_gather_shape(tensors_: Tuple[torch.Tensor]) -> List[torch.Tensor]:
    # import ipdb;ipdb.set_trace()
    tensor = torch.cat(tensors_)
    tensors = [torch.zeros(1,device = tensor.device)[0] for _ in range(todd.base.get_world_size())]
    # print(todd.get_rank(),tensors,torch.tensor(tensor.shape[0],device = tensor.device))
    torch.distributed.all_gather(tensors,torch.tensor(tensor.shape[0],device = tensor.device,dtype=tensors[0].dtype))
    return tensors

class CocoClassification(torchvision.datasets.CocoDetection):
    _classnames: List[str]
    _cat2label: Dict[int, int]

    def __init__(
        self,
        root: str,
        ann_file: str,
        pretrained: str,
        proposal: Optional[str] = None,
        split: Optional[str] = None,
        lvis_ann_file:Optional[str] = None,
        lvis_split: Optional[str] = None,
        top_KP :int = 100,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )
        self._ann_file = ann_file
        self.top_KP = top_KP
        if split is None:
            self._classnames = [cat['name'] for cat in self.coco.cats.values()]
            self._cat2label = {cat: i for i, cat in enumerate(self.coco.cats)}
        else:
            classnames = getattr(datasets, lvis_split)
            novel_classnames = getattr(datasets, 'LVIS_337')
            self._lvis = LVIS(lvis_ann_file)
            self._classnames =[]
            self._novelclassnames = []
            self._novelids = []
            self._cat2label = dict()
            # import ipdb;ipdb.set_trace()
            for cat in self._lvis.cats.values():
                if cat['name'] in classnames:
                    self._classnames.append(cat['name'])
                    self._cat2label[cat['id']] = len(self._cat2label)
                if cat['name'] in novel_classnames:
                    self._novelclassnames.append(cat['name'])
                    self._novelids.append(self._cat2label[cat['id']])
            # import ipdb;ipdb.set_trace()
            self._mask = torch.ones(len(self._classnames))
            # self._mask[self._novelids] = 1*(-10000000000000)

            self._label2cat = dict()
            for cat_id in self._cat2label.keys():
                self._label2cat[self._cat2label[cat_id]] = cat_id
        # import ipdb;ipdb.set_trace()
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
            self._scaler  = 1
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

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [anno for anno in target if anno['category_id'] in self._cat2label]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor]:
        data = super().__getitem__(index)
        image,target = data
        image_id = torch.tensor(self.ids[index])
        proposal_pth = f'{image_id.item():012d}.pth'
        data_path = os.path.join(self.proposal_root,proposal_pth)
        data_ckpt = torch.load(data_path, 'cpu')
        
        if 'proposal_embeddings' in data_ckpt.keys():# my generate version
            proposal_embeddings = data_ckpt['proposal_embeddings']
            proposal_objectness = data_ckpt['proposal_objectness']
            proposal_bboxes = data_ckpt['proposal_bboxes']
        elif 'objectness' in data_ckpt.keys():# lt generate version
            proposal_bboxes = data_ckpt['bboxes']
            proposal_embeddings = data_ckpt['patches']
            proposal_objectness = data_ckpt['objectness']
        elif len(data_ckpt.keys()) ==2:
            proposal_bboxes = data_ckpt['bboxes']
            proposal_embeddings = data_ckpt['patches']
            proposal_objectness= torch.ones_like(proposal_bboxes[:,-1])
        else:
            raiseExceptions("No such data format")    
        
        inds = torch.arange(self.top_KP)
        return proposal_embeddings[inds],proposal_objectness[inds],proposal_bboxes[inds],image_id

    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
        
        proposal_embeddings, proposal_objectness, proposal_bboxes,image_ids = map(torch.stack, zip(*batch))
        return Batch(proposal_embeddings, proposal_objectness, proposal_bboxes,image_ids,self._class_embeddings, self._scaler, self._bias,self._mask)


class Model(todd.base.Module):

    def __init__(
        self,
        *args,
        config: todd.base.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model, _ = clip.load(config.pretrained, 'cpu')
        assert "ViT-B" in config.pretrained and "32" in config.pretrained
        self.model.requires_grad_(False)
        self.softmax_t = config.softmax_t
        self.topK_clip_scores = config.topK_clip_scores
        self.nms_score_thres = config.nms_score_thres
        self.nms_iou_thres =  config.nms_iou_thres
        self.score_fusion_cfg = config.bbox_objectness



    def forward(self, batch: Batch) -> torch.Tensor:
        # import ipdb;ipdb.set_trace()
        proposal_embeddings = batch.proposal_embeddings
        proposal_objectness = batch.proposal_objectness
        assert proposal_embeddings.shape[1] == proposal_objectness.shape[1]
        proposal_embeddings = proposal_embeddings / \
        proposal_embeddings.norm(dim=2, keepdim=True)

        clip_logit = (proposal_embeddings @ batch.class_embeddings.T)
        # import ipdb;ipdb.set_trace()
        clip_logit = clip_logit*batch.mask[None,None]
        clip_logit = (1/self.softmax_t)*clip_logit
        clip_logit = torch.softmax(clip_logit,dim = 2)
        clip_logit_v,clip_logit_i = torch.topk(clip_logit,self.topK_clip_scores,dim = 2)
        # import ipdb;ipdb.set_trace()
        clip_logit_k = clip_logit * (clip_logit>=clip_logit_v[...,-1:])   
        
        # fusion
        if self.score_fusion_cfg['_name'] == 'add':
            final_logit_k = (clip_logit_k*self.score_fusion_cfg['clip_score_ratio']) + ((clip_logit_k>0)*batch.proposal_objectness[...,None]*self.score_fusion_cfg['obj_score_ratio'])
        elif self.score_fusion_cfg['_name'] == 'mul' :
            final_logit_k = (clip_logit_k**self.score_fusion_cfg['clip_score_ratio']) * (batch.proposal_objectness[...,None]**self.score_fusion_cfg['obj_score_ratio'])
        else:
            raise ValueError(self.score_fusion_cfg['_name'])
        
        # split batch to each image to nms/thresh
        final_bboxes = []
        final_labels = []
        final_image = []
        # import ipdb;ipdb.set_trace()
        for i,(result,logit) in enumerate(zip(batch.proposal_bboxes,final_logit_k)):
            # assert (result[:,3]<result[:,1]).any()
            final_bbox_c, final_label = multiclass_nms(result[...,:4].float(),logit.float(),score_thr=self.nms_score_thres,nms_cfg=dict(type='nms', iou_threshold=self.nms_iou_thres))
            # final_bbox = torch.cat((final_bbox_c,final_label[...,None]),dim = -1) 
            # import ipdb;ipdb.set_trace()
            image_ids = batch.image_ids[i].repeat(final_bbox_c.shape[0])
            # final_bbox = torch.cat((final_bbox,image_ids),dim = -1)
            # assert (final_bbox_c[:,3]<final_bbox_c[:,1]).any()
            final_bboxes.append(final_bbox_c)
            final_labels.append(final_label) 
            final_image.append(image_ids) 
        final_bboxes = torch.cat(final_bboxes)  
        final_labels= torch.cat(final_labels)  
        final_image = torch.cat(final_image)  
        return final_bboxes,final_labels,final_image


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
        # import ipdb;ipdb.set_trace()
        sampler = (
            None if (not config.sample) or debug.CPU else
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
        # import ipdb;ipdb.set_trace()
        if not debug.CPU and config.dis :
            model = torch.nn.parallel.DistributedDataParallel(
                model.cuda(),
                device_ids=[torch.cuda.current_device()],
            )
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() if isinstance(x, torch.Tensor) else x for x in batch])
        final_bboxes,final_labels,final_image = self._model(batch)
        memo['results'].append((final_bboxes,final_labels,final_image))


    def _after_run_iter(self, *args, i: int, batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> None:
        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if self._config.model.dis:
            rank,world_size = get_dist_info()
            self._logger.info(rank)
            shapes = tuple(
                    map(all_gather_shape,results),
            )

            max_shapes = [max(shapes[0]),max(shapes[0]),max(shapes[0])]
            results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
            results = tuple(  # all_gather must exec for all ranks ,may result in cuda out of memory
                    map(all_gather, results, max_shapes),
                )
            self._logger.info(str(rank)+" Gather Done")
            if rank != 0:
                return None
        
            bboxes_, labels_, images_ = results
            bboxes = []
            labels = []
            images = []
            for i,(bbox,label,image) in enumerate(zip(bboxes_, labels_, images_)):
                ind = torch.arange(shapes[0][i].int())
                bboxes.append(bbox[ind])
                labels.append(label[ind])
                images.append(image[ind])
        else:
            bboxes, labels, images = results
            
        bboxes = torch.cat(bboxes).tolist()
        labels = torch.cat(labels).tolist()
        images = torch.cat(images).tolist()
        lvisGt:LVIS = self._dataset._lvis
        new_annotations = list()
        imageId_list=list()
        for currbox,label,image_id in zip(bboxes, labels, images):
            if image_id not in lvisGt.imgs.keys():
                    continue
            x0, y0, x1, y1 = currbox[:4]       # xyxy
            box = [x0, y0, x1 - x0, y1 - y0]    # xywh
            if (x1-x0 <=0) or (y1-y0<=0):
                continue
            curConf = currbox[-1]
            catId_top1 = label

            data = {'image_id': image_id,
                    'category_id': self._dataset._label2cat[catId_top1],
                    'bbox': box,
                    'score': curConf}
            new_annotations.append(data)
            imageId_list.append(image_id)
        self._logger.info( 'Total PL boxes num: %d, avg num: %.2f' % (len(new_annotations), len(new_annotations)/len(set(imageId_list))) )
        lvisDt = LVISResults(lvisGt,new_annotations)
        self._logger.info("Begin Evaluation")
        lvisEval = LVISEval(lvisGt, lvisDt, iou_type='bbox')
        lvisEval.run()
        results = lvisEval.get_results()
        lvisEval.print_results()
        nni.report_final_result(results['AP50'])
        self._logger.info("END")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    # parser.add_argument('root', type=pathlib.Path)
    parser.add_argument('--hotwater', action='store_true')
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    params = nni.get_next_parameter()
    # import ipdb;ipdb.set_trace = lambda:0
    if len(params) == 0:
        params = dict(
            top_KP = 150,
            softmax_t = 0.01,
            topK_clip_scores = 1,
            # nms_score_thres = 0.0,
            nms_iou_thres = 0.6,
            bbox_objectness=dict(
                _name='add',
                clip_score_ratio= 0.5,
                obj_score_ratio = 0.8,

            ),
        )
    print(params)
    # config = todd.base.Config.load(args.config)
    data_root = 'data/coco/'
    config = todd.Config(
        val = dict(
            dataloader=dict(
                batch_size=8,
                num_workers=8,
                sample = not args.hotwater,
                dataset=dict(
                    root=data_root+"train2017",
                    ann_file=data_root+'annotations/instances_train2017.json',
                    pretrained='data/coco/prompt/prompt2.pth',
                    split='COCO_17',
                    proposal = 'data/coco/mask_embeddings/train',
                    top_KP = params['top_KP'],
                    lvis_ann_file='data/lvis_v1/annotations/lvis_v1_train.json',
                    lvis_split='LVIS'
                    
            ),
        )),
        logger=dict(
            nterval=64,
        ),
        
        model = dict(
            dis = not args.hotwater,
            pretrained = 'pretrained/clip/ViT-B-32.pt',
            softmax_t = params['softmax_t'],
            # softmax_t = 1,
            topK_clip_scores = params['topK_clip_scores'],
            nms_score_thres = 0.0,
            nms_iou_thres = params['nms_iou_thres'],
            bbox_objectness = params['bbox_objectness']
        ),
        
    )
    if args.odps is not None:
        odps_init(args.odps)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if not debug.CPU and not args.hotwater:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    runner = Runner(name=args.name, config=config)
    runner.run()
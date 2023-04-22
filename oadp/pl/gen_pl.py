import argparse
import json
import os
import pathlib
from typing import Iterator, List, Optional, Tuple

import todd
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.dataloader
import torch.utils.data.distributed
from lvis.lvis import LVIS
from mmcv.runner import get_dist_info
from pycocotools.coco import COCO
from todd.utils import Memo

from ..base import globals_ as cats
from .datasets import Batch, PseudoLabelDataset
from .model import build_model


def all_gather(tensors_: Tuple[torch.Tensor],
               shape: torch.Tensor) -> List[torch.Tensor]:
    tensor = torch.cat(tensors_)
    tensors = []
    for _ in range(todd.base.get_world_size()):
        if len(tensor.shape) == 2:
            tensors.append(
                torch.zeros([int(shape), tensor.shape[1]],
                            device=tensor.device,
                            dtype=tensor.dtype)
            )
        else:
            tensors.append(
                torch.zeros([int(shape)],
                            device=tensor.device,
                            dtype=tensor.dtype)
            )
    if len(tensor.shape) == 2:
        fake_tensor = torch.zeros([
            int(shape) - tensor.shape[0], tensor.shape[1]
        ],
                                  device=tensor.device,
                                  dtype=tensor.dtype)
    else:
        fake_tensor = torch.zeros([int(shape) - tensor.shape[0]],
                                  device=tensor.device,
                                  dtype=tensor.dtype)
    tensor = torch.cat((tensor, fake_tensor))
    torch.distributed.all_gather(tensors, tensor)
    return tensors


def all_gather_shape(tensors_: Tuple[torch.Tensor]) -> List[torch.Tensor]:
    tensor = torch.cat(tensors_)
    tensors = [
        torch.zeros(1, device=tensor.device)[0]
        for _ in range(todd.base.get_world_size())
    ]
    torch.distributed.all_gather(
        tensors,
        torch.tensor(
            tensor.shape[0], device=tensor.device, dtype=tensors[0].dtype
        )
    )
    return tensors


class Runner(todd.utils.BaseRunner):

    def __init__(self, *args, annotation, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.annotation: todd.Config = annotation

    def _build_dataloader(
        self,
        config: Optional[todd.base.Config],
    ) -> Tuple[PseudoLabelDataset,
               Optional[torch.utils.data.distributed.DistributedSampler],
               torch.utils.data.DataLoader,
               ]:
        assert config is not None
        self._dataset = dataset = PseudoLabelDataset(**config.dataset)
        sampler = (
            None if (not config.sample) or todd.Store.CPU else
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
        return dataloader

    def _before_run(self) -> Memo:
        memo = super()._before_run()
        memo.update(results=[])
        return memo

    def _run_iter(self, batch, memo: Memo) -> torch.Tensor:
        if not todd.Store.CPU:
            batch = Batch(
                *[
                    x.cuda() if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]
            )
        self.model.half()
        final_bboxes, final_labels, final_image = self._model(batch)
        memo['results'].append((final_bboxes, final_labels, final_image))
        return torch.tensor(0.0)

    def _init_dataset(self):
        if self.annotation.type == "coco_novel":
            self.json_func = self.coco_novel
        elif self.annotation.type == "lvis":
            self.json_func = self.lvis_to_coco
        else:
            raise RuntimeError("no such pl type")

    def _after_run(self, memo: Memo) -> None:
        self._init_dataset()
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if self.model.dist:
            rank, _ = get_dist_info()
            self._logger.info(rank)
            shapes = tuple(map(all_gather_shape, results), )

            max_shapes = [max(shapes[0]), max(shapes[0]), max(shapes[0])]
            results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
            results = tuple(  # all_gather must exec for all ranks ,may result in cuda out of memory
                    map(all_gather, results, max_shapes),
            )
            self._logger.info(str(rank) + " Gather Done")
            if rank != 0:
                return None

            bboxes_, labels_, images_ = results
            bboxes = []
            labels = []
            images = []
            for i, (bbox, label,
                    image) in enumerate(zip(bboxes_, labels_, images_)):
                ind = torch.arange(shapes[0][i].int())
                bboxes.append(bbox[ind])
                labels.append(label[ind])
                images.append(image[ind])
        else:
            bboxes, labels, images = results

        bboxes = torch.cat(bboxes).tolist()
        labels = torch.cat(labels).tolist()
        images = torch.cat(images).tolist()

        new_annotations = list()
        imageId_list = list()
        for currbox, label, image_id in zip(bboxes, labels, images):
            x0, y0, x1, y1 = currbox[:4]  # xyxy
            box = [x0, y0, x1 - x0, y1 - y0]  # xywh
            if (x1 - x0 <= 0) or (y1 - y0 <= 0):
                continue

            curConf = currbox[-1]
            catId_top1 = label

            data = {
                'image_id': image_id,
                'category_id': self._dataset._label2cat[catId_top1],
                'bbox': box,
                'score': curConf,
            }

            new_annotations.append(data)
            imageId_list.append(image_id)
        self._logger.info('Total image num: %d' % (len(set(imageId_list))))
        self._logger.info(
            'Total PL boxes num: %d, avg num: %.2f' % (
                len(new_annotations),
                len(new_annotations) / len(set(imageId_list))
            )
        )
        self.json_func(new_annotations)

    def lvis_to_coco(self, new_annotations) -> None:
        lvisGt: LVIS = self._dataset._lvis
        lvis_data = lvisGt.dataset
        coco_data = self._dataset.coco.dataset
        lvis_categories = lvisGt.dataset['categories']
        new_categories = []
        name2cat = dict()
        for cat in lvis_categories:
            name2cat[cat['name']] = cat
        lvis_names = getattr(cats, 'LVIS')

        for i, name in enumerate(lvis_names):
            cat = name2cat[name]
            cat['id'] = i
            new_categories.append(cat)
        lvis_data['categories'] = new_categories

        coco_name = getattr(cats, 'COCO_48_17')
        new_pl = 0
        new_pls = []
        for i, data in enumerate(new_annotations):
            x0, y0, w, h = data['bbox']
            y1 = y0 + h
            x1 = x0 + w
            class_name = self._dataset._classnames[self._dataset._cat2label[
                data['category_id']]]
            if any(coco_n in class_name for coco_n in coco_name):
                continue

            new_pl += 1
            data['id'] = (i + 1)
            data['segmentation'] = [[x0, y0, x0, y1, x1, y1, x1, y0]]
            data['area'] = w * h
            data['category_id'] = lvis_names.index(class_name)
            new_pls.append(data)

        self._logger.info("Generate new pl: " + str(new_pl))
        lvis_data['annotations'] = new_pls
        lvis_data['images'] = coco_data['images']
        json_path = os.path.join(self._work_dir, self.annotation.name)
        with open(json_path, 'w') as f:
            json.dump(lvis_data, f)
        self._logger.info("Save END")

    def coco_novel(self, new_annotations) -> None:
        cocoGt = COCO(self._dataset._ann_file)
        coco_data = cocoGt.dataset
        coco_categories = coco_data['categories']
        coco_anno = coco_data['annotations']
        new_categories = []
        name2cat = dict()
        id2name = dict()
        for cat in coco_categories:
            name2cat[cat['name']] = cat
            id2name[cat['id']] = cat['name']
        coco_names = getattr(cats, 'COCO_48_17')
        coco_names_base = getattr(cats, 'COCO_48')
        for i, name in enumerate(coco_names):
            cat = name2cat[name]
            cat['id'] = i
            new_categories.append(cat)

        new_coco_anno = list()
        anno_id = 1

        for anno in coco_anno:
            class_name = id2name[anno['category_id']]
            if class_name not in coco_names_base:
                continue
            else:
                anno['category_id'] = coco_names.index(class_name)
                anno['id'] = anno_id
                anno_id += 1
                new_coco_anno.append(anno)
        new_id = 0
        for i, data in enumerate(new_annotations):
            x0, y0, w, h = data['bbox']
            y1 = y0 + h
            x1 = x0 + w
            data['id'] = (anno_id + 1)
            anno_id += 1
            data['segmentation'] = [[x0, y0, x0, y1, x1, y1, x1, y0]]
            data['area'] = w * h
            class_name = id2name[data['category_id']]
            if class_name in coco_names_base:
                continue
            else:
                new_id += 1
            data['category_id'] = coco_names.index(class_name)
            new_coco_anno.append(data)

        self._logger.info("New generate " + str(new_id) + " novel PL")
        coco_data['annotations'] = new_coco_anno
        coco_data['categories'] = new_categories
        json_path = os.path.join(self._work_dir, self.annotation.name)
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)
        self._logger.info("Save END")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--override', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.base.Config.load(args.config)

    if args.override is not None:
        for k, v in args.override.items():
            todd.base.setattr_recur(config, k, v)

    if todd.Store.CUDA:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.get_local_rank())

    model = build_model(config.model)
    runner = Runner(
        name=args.name,
        model=model,
        dataloader=config.dataloader,
        log=config.logger,
        load_state_dict=todd.Config(),
        state_dict=todd.Config(),
        annotation=config.annotation_file
    )
    runner.run()

import argparse
import json
import os
import pathlib

import todd
import torch
import torch.distributed
import torch.utils.data
from mmcv.runner import get_dist_info
from mmdet.core import multiclass_nms
from todd.utils import Memo
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..base import globals_
from .datasets import Batch, Dataset
from .utils import all_gather, all_gather_shape


class Runner(todd.utils.BaseRunner):

    def __init__(self, *args, generator, json_file, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generator: todd.Config = generator
        self.json_file: todd.Config = json_file

    def _build_dataloader(self, config: todd.base.Config) -> DataLoader:
        assert config is not None
        self._dataset = dataset = Dataset(**config.dataset)
        sampler: DistributedSampler = (
            DistributedSampler(
                dataset,
                shuffle=False,
            )
        )
        dataloader: DataLoader = DataLoader(
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

    def generat_pl(self, batch: Batch) -> tuple[torch.Tensor, ...]:
        # normalization
        embeddings = batch.proposal_embeddings
        objectness = batch.proposal_objectness
        embeddings_norm = embeddings.norm(dim=2, keepdim=True)
        embeddings = embeddings / embeddings_norm
        # cosine similarity
        clip_logit = embeddings @ batch.class_embeddings.T
        clip_logit = clip_logit * batch.scaler - batch.bias
        clip_logit = (1 / self.generator.softmax_t) * clip_logit
        clip_logit = torch.softmax(clip_logit, dim=2)

        clip_logit_v, _ = torch.topk(
            clip_logit, self.generator.topK_clip_scores, dim=2
        )
        clip_logit_k = clip_logit * (clip_logit >= clip_logit_v[..., -1:])
        # fusion
        clip_score = clip_logit_k**self.generator.clip_score_ratio
        obj_score = objectness**self.generator.obj_score_ratio
        final_logit_k = clip_score * obj_score
        # split batch to each image to nms/thresh
        final_bboxes = []
        final_labels = []
        final_image = []
        for i, (result, logit) in enumerate(
            zip(batch.proposal_bboxes, final_logit_k)
        ):
            final_bbox_c, final_label = multiclass_nms(
                result[..., :4].float(),
                logit.float(),
                score_thr=self.generator.nms_score_thres,
                nms_cfg=dict(
                    type='nms', iou_threshold=self.generator.nms_iou_thres
                )
            )
            image_ids = batch.image_ids[i].repeat(final_bbox_c.shape[0])
            final_bboxes.append(final_bbox_c)
            final_labels.append(final_label)
            final_image.append(image_ids)

        tensor_final_bboxes = torch.cat(final_bboxes)
        tensor_final_labels = torch.cat(final_labels)
        tensor_final_image = torch.cat(final_image)
        return tensor_final_bboxes, tensor_final_labels, tensor_final_image

    def _run_iter(self, batch, memo: Memo) -> torch.Tensor:
        if not todd.Store.CPU:
            batch = Batch(
                *[
                    x.cuda() if isinstance(x, torch.Tensor) else x
                    for x in batch
                ]
            )

        final_bboxes, final_labels, final_image = self.generat_pl(batch)
        memo['results'].append((final_bboxes, final_labels, final_image))
        return torch.tensor(0.0)

    def gather(self, memo: Memo) -> tuple[list, ...]:
        results = zip(*memo['results'])

        if torch.cuda.device_count() > 1:
            rank, _ = get_dist_info()
            self._logger.info(rank)
            shapes = tuple(map(all_gather_shape, results))
            max_shapes = torch.Tensor([
                max([shape.item() for shape in shapes[0]]) for _ in range(3)
            ])
            gather_results = tuple(map(all_gather, results, max_shapes))
            self._logger.info(str(rank) + " Gather Done")

            bboxes_, labels_, images_ = gather_results
            bboxes, labels, images = [], [], []
            for i, (bbox, label,
                    image) in enumerate(zip(bboxes_, labels_, images_)):
                ind = torch.arange(int(shapes[0][i].int()))
                bboxes.append(bbox[ind])
                labels.append(label[ind])
                images.append(image[ind])
        else:
            bboxes, labels, images = [list(res) for res in results]

        bboxes = torch.cat(bboxes).tolist()
        labels = torch.cat(labels).tolist()
        images = torch.cat(images).tolist()

        return bboxes, labels, images

    def _after_run(self, memo: Memo) -> None:
        # gather the result
        bboxes, labels, images = self.gather(memo)
        # annotations in coco format
        # load the coco ann
        coco_ann = self._dataset.coco.dataset
        new_annotations, image_id_list = [], []
        for currbox, label, image_id in zip(bboxes, labels, images):
            x0, y0, x1, y1, score = currbox
            box = [x0, y0, x1 - x0, y1 - y0]  # xywh
            if (x1 - x0 <= 0) or (y1 - y0 <= 0):
                continue
            data = {
                'image_id': image_id,
                'category_id': int(label),
                'bbox': box,
                'score': score,
                'segmentation': [[x0, y0, x0, y1, x1, y1, x1, y0]],
                'area': (x1 - x0) * (y1 - y0)
            }
            new_annotations.append(data)
            image_id_list.append(image_id)
        # log the PL info
        self._logger.info('Total image num: %d' % (len(set(image_id_list))))
        self._logger.info(
            'Total PL boxes num: %d, avg num: %.2f' % (
                len(new_annotations),
                len(new_annotations) / len(set(image_id_list))
            )
        )

        if self.json_file.type == "coco":
            coco_ann['annotations'] += new_annotations
        if self.json_file.type == "lvis":
            coco_name = globals_.coco.all_
            lvis_annotations = []
            for ann in new_annotations:
                class_name = globals_.lvis.all_[ann['category_id']]
                if any(coco_n in class_name for coco_n in coco_name):
                    continue
                lvis_annotations.append(ann)
            coco_ann['annotations'] = lvis_annotations
            coco_ann['categories'] = globals_.lvis.all_
        # save
        path = os.path.join(self._work_dir, self.json_file.name)
        with open(path, 'w') as f:
            json.dump(coco_ann, f)


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
    print(config)
    runner = Runner(
        name=args.name,
        model=None,
        dataloader=config.dataloader,
        log=config.logger,
        load_state_dict=todd.Config(),
        state_dict=todd.Config(),
        generator=config.generator,
        json_file=config.json_file
    )
    runner.run()

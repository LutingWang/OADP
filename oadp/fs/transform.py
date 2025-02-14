import os
import json
import re
import random
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from mmcv.transforms import BaseTransform
from mmengine.structures import BaseDataElement
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import BaseBoxes
from mmengine.registry import FUNCTIONS

BICUBIC = InterpolationMode.BICUBIC

@FUNCTIONS.register_module()
def fs_collect_fn(data_batch: list):
    batched_inputs, batched_texts, batched_shots = [], [], []
    for data in data_batch:
        batched_inputs.extend(data['inputs'])
        batched_texts.append(data['texts'])
        batched_shots.append(data['shots'])
    return {
        "inputs": torch.stack(batched_inputs, dim=0),
        "texts": batched_texts,
        "shots": batched_shots
    }



@TRANSFORMS.register_module()
class ReplaceLabel(BaseTransform):
    def __init__(self, label_map_path: str) -> None:
        self._label_map = tuple(json.load(open(label_map_path)))

    def transform(self, results: dict) -> dict:
        results['text'] = self._label_map
        return results

@TRANSFORMS.register_module()
class RandomLoadFromFile(BaseTransform):
    def __init__(self, min_imgs: int, max_imgs: int) -> None:
        self.min_imgs = min_imgs
        self.max_imgs = max_imgs

    def transform(self, results: dict) -> dict:
        image_num = len(results['img_path'])
        min_imgs = min(self.min_imgs, image_num)
        max_imgs = min(self.max_imgs, image_num)
        num = random.randint(min_imgs, max_imgs)
        imgs_path = random.choices(results['img_path'], k=num)
        imgs = []
        for img_path in imgs_path:
            imgs.append(Image.open(img_path))
        results['img'] = imgs
        results['shots'] = len(imgs)
        return results


@TRANSFORMS.register_module()
class ClipTransform(BaseTransform):
    def __init__(self, in_key = 'img', out_key = 'inputs', n_px = 224) -> None:
        self.in_key = in_key
        self.out_key = out_key
        self._clip_transform = Compose([
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            self._convert_image_to_rgb,
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), 
                      (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _convert_image_to_rgb(self, image):
        return image.convert("RGB")

    def transform(self, results: dict) -> dict:
        inputs = []
        for img in results[self.in_key]:
            inputs.append(self._clip_transform(img))
        results[self.out_key] = inputs
        return results

@TRANSFORMS.register_module()
class CleanText(BaseTransform):
    @staticmethod
    def clean_label_name(name: str) -> str:
        name = re.sub(r'\(.*\)', '', name)
        name = re.sub(r'_', ' ', name)
        name = re.sub(r'  ', ' ', name)
        return name
    
    def transform(self, results: dict) -> dict:
        results['text'] = self.clean_label_name(results['text'])
        return results


@TRANSFORMS.register_module()
class PackFsData(BaseTransform):
    def transform(self, results: dict) -> dict:
        return {
            "inputs": results['inputs'],
            "texts": results['text'],
            "shots": results['shots']
        }


@TRANSFORMS.register_module()
class InsertLvisFsInputs(BaseTransform):
    def __init__(self, exemplar_path: str) -> None:
        exemplar = json.load(open(exemplar_path, "r"))
        self.exemplar_dict = exemplar


@TRANSFORMS.register_module()
class SampleRefImages(BaseTransform):
    def __init__(self, 
        min_imgs: int,
        max_imgs: int,
        max_sample_num: int, 
        label_map_path: str,
        samples_data_root: str,
        samples_label_map: str,
        sample_negatives: bool = True
    ) -> None:
        self.min_imgs = min_imgs
        self.max_imgs = max_imgs
        self.max_sample_num = max_sample_num
        self.samples_data_root = samples_data_root
        self.sample_negatives = sample_negatives
        self.samples_label_map = json.load(open(samples_label_map, "r")) # samples_id -> image_path
        self.label_map = json.load(open(label_map_path, "r")) # cate_id -> samples_id

    def sample_ref_images(self, labels) -> list:
        n_images = random.randint(self.min_imgs, self.max_imgs)
        ref_images = []
        text = []
        for label in labels:
            label_images =self.samples_label_map[self.label_map[label]["samples_id"]]
            cate_name = self.label_map[label]["name"]
            random_images = random.choices(label_images, k=n_images)
            images = [Image.open(os.path.join(self.samples_data_root, image_path)) 
                      for image_path in random_images]
            ref_images.extend(images)
            text.append(cate_name)
        return ref_images, n_images, text

    def sample_num_samples(self, gt_bboxes, gt_labels):
        # get positive labels
        positive_labels = set([str(label) for label in gt_labels])
        # get nagative labels
        if len(positive_labels) < self.max_sample_num:
            nagative_labels = list(set(self.label_map.keys()) - positive_labels)
            assert self.max_sample_num < len(nagative_labels)
            nagative_sample_num = self.max_sample_num - len(positive_labels)
            nagative_labels = random.choices(nagative_labels, k=nagative_sample_num)
            kept_gt_labels = gt_labels
            vaild_positive_labels = list(positive_labels)
        else:
            # random choose positive labels
            kept_positive_labels = random.choices(list(positive_labels), k=self.max_sample_num)
            keep_box_index = []
            kept_gt_labels = []
            for i, gt_label in enumerate(gt_labels):
                if gt_label in kept_positive_labels:
                    keep_box_index.append(i)
                    kept_gt_labels.append(gt_label)
            
            nagative_labels = []
            vaild_positive_labels = kept_positive_labels
            gt_bboxes = gt_bboxes[keep_box_index]

        return kept_gt_labels, vaild_positive_labels + nagative_labels, gt_bboxes
        

    def shuffle_reindex(self, labels, gt_labels):
        random.shuffle(labels)
        positive_maps = torch.zeros((len(gt_labels), self.max_sample_num), dtype=torch.float32)
        
        # get the positive label index
        label_remap_dict = {}
        for i, pos_label in enumerate(gt_labels):
            for j, label in enumerate(labels):
                if int(label) == pos_label:
                    positive_maps[i, j] = 1
                    label_remap_dict[int(label)] = j
                    break
        if len(gt_labels) > 0:
            gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)
        return positive_maps, gt_labels

    def transform(self, results: dict) -> dict:
        # get gt_boxes and gt_labels
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        if self.sample_negatives:
            # get sampled labels
            gt_labels, sampled_labels, gt_bboxes = self.sample_num_samples(gt_bboxes, gt_labels)
            # shuffle and reindex
            positive_maps, gt_labels = self.shuffle_reindex(sampled_labels, gt_labels)
        else:
            positive_maps = None
            sampled_labels = gt_labels
        # sample images
        ref_images, n_images, text = self.sample_ref_images(sampled_labels)
        # add info to results
        results['ref_images'] = ref_images
        results['n_images'] = n_images
        results['n_samples'] = self.max_sample_num
        results['text'] = text
        results['tokens_positive'] = positive_maps
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        return results
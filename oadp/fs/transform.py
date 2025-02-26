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
        label_map_path: str,
        samples_data_root: str,
        samples_label_map: str,
        max_sample_num: int|None = None,
        sample_feature: bool = True
    ) -> None:
        self.min_imgs = min_imgs
        self.max_imgs = max_imgs
        self.max_sample_num = max_sample_num
        self.samples_data_root = samples_data_root
        self.sample_feature = sample_feature
        self.samples_label_map = json.load(open(samples_label_map, "r")) # samples_id -> image_path
        self.label_map = json.load(open(label_map_path, "r")) # cate_id -> samples_id

    def sample_ref_images_feature(self, labels) -> list:
        n_shots = random.randint(self.min_imgs, self.max_imgs)
        ref_images = []
        text = []
        for label in labels:
            sample_id = self.label_map[label]["samples_id"]
            cate_name = self.label_map[label]["name"]
            features_path = f"{os.path.join(self.samples_data_root, sample_id)}.pth"
            features = torch.load(features_path, map_location='cpu')
            clip_features = features["clip_features"]
            dino_features = features["dino_features"]
            cat_features = torch.cat([clip_features, dino_features], dim=1)
            image_ids = torch.randint(0, cat_features.size(0), (n_shots,))
            random_features = cat_features[image_ids]
            ref_images.extend(random_features)
            text.append(cate_name)
        return ref_images, n_shots, text


    def sample_ref_images(self, labels) -> list:
        n_shots = random.randint(self.min_imgs, self.max_imgs)
        ref_images = []
        text = []
        for label in labels:
            label_images =self.samples_label_map[self.label_map[label]["samples_id"]]
            cate_name = self.label_map[label]["name"]
            random_images = random.choices(label_images, k=n_shots)
            images = [Image.open(os.path.join(self.samples_data_root, image_path)) 
                      for image_path in random_images]
            ref_images.extend(images)
            text.append(cate_name)
        return ref_images, n_shots, text

    def sample_num_samples(self, gt_bboxes, gt_labels):
        # get positive labels
        positive_labels = set([str(label) for label in gt_labels])
        # get nagative labels
        if len(positive_labels) <= self.max_sample_num:
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
            j = labels.index(str(pos_label))
            positive_maps[i, j] = 1
            label_remap_dict[pos_label] = j

        assert len(gt_labels) > 0  
        gt_labels = np.vectorize(lambda x: label_remap_dict[x])(gt_labels)
        return positive_maps, gt_labels

    def transform(self, results: dict) -> dict:
        # get gt_boxes and gt_labels
        gt_bboxes = results['gt_bboxes']
        if isinstance(gt_bboxes, BaseBoxes):
            gt_bboxes = gt_bboxes.tensor
        gt_labels = results['gt_bboxes_labels']
        if self.max_sample_num is not None:
            # get sampled labels
            gt_labels, sampled_labels, gt_bboxes = self.sample_num_samples(gt_bboxes, gt_labels)
            # shuffle and reindex
            positive_maps, gt_labels = self.shuffle_reindex(sampled_labels, gt_labels)
        else:
            positive_maps = None
            sampled_labels = list(self.label_map.keys())
            self.max_sample_num = len(self.label_map)
        # sample images
        if self.sample_feature:
            ref_images, n_shots, text = self.sample_ref_images_feature(sampled_labels)
        else:
            ref_images, n_shots, text = self.sample_ref_images(sampled_labels)
        # add info to results
        results['ref_images'] = ref_images
        results['n_shots'] = n_shots
        results['n_samples'] = self.max_sample_num
        results['text'] = text
        results['tokens_positive'] = positive_maps
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        return results


@TRANSFORMS.register_module()
class SampleRefImagesVG(BaseTransform):
    def __init__(self, 
        min_imgs: int,
        max_imgs: int,
        label_map_path: str,
        samples_data_root: str,
        num_classes: int = 256,
        training: bool = True
    ) -> None:
        self.min_imgs = min_imgs
        self.max_imgs = max_imgs
        self.num_classes = num_classes
        self.samples_data_root = samples_data_root
        self.label_map = json.load(open(os.path.join(samples_data_root, label_map_path), "r"))
        self.training = training

    def sample_ref_images_feature(self, labels: list[str]) -> list:
        n_shots = random.randint(self.min_imgs, self.max_imgs)
        ref_images = []
        for label in labels:
            sample_paths = list(self.label_map[label].keys())
            # Randomly sample n_shots paths from sample_paths
            assert n_shots <= len(sample_paths)
            sampled_paths = random.sample(sample_paths, n_shots)
            cat_features = []
            for sample_path in sampled_paths:
                # Use pth_helprt to load the features from the sample path
                features = self.pth_helper(sample_path)
                cat_features.append(features)
            ref_images.append(torch.cat(cat_features, dim=0))
        return ref_images, n_shots
    
    def pth_helper(self, image_path: str):
        # Assume image_path is 'data/imagenet-21k/n11908549/n11908549_2429.JPEG'
        # We extract the folder name and image id.
        base_name = os.path.basename(image_path)            # n11908549_2429.JPEG
        folder_name = os.path.basename(os.path.dirname(image_path))  # n11908549
        image_id = os.path.splitext(base_name)[0]             # n11908549_2429

        # Build the path to the .pth file, e.g., 'data/imagenet-21k/n11908549/n11908549.pth'
        pth_path = os.path.join(self.samples_data_root, "features", f"{folder_name}.pth")
        # Load the .pth file features
        features = torch.load(pth_path, map_location='cpu')
        index = features['ids'].index(image_id)
        clip_features = features['clip_features'][index].unsqueeze(0)
        dino_features = features['dino_features'][index].unsqueeze(0)
        return torch.cat([clip_features, dino_features], dim=1)
    
    def get_positive_map(self, gt_labels: torch.Tensor):
        positive_maps = torch.zeros((len(gt_labels), self.num_classes), dtype=torch.float32)
        for i, pos_label in enumerate(gt_labels):
            positive_maps[i, i] = 1
        return positive_maps

    def transform(self, results: dict) -> dict:
        # get gt_boxes and gt_labels
        gt_bboxes = results['gt_bboxes']
        gt_labels = results['gt_bboxes_labels']
        phrases_dict = results['phrases']
        # sample labels
        sampled_labels = []
        sorted_gt_labels = sorted(set(gt_labels))
        for label in sorted_gt_labels:
            label_elem = phrases_dict[label]['phrase']
            if isinstance(label_elem, list):
                sampled_labels.append(random.choice(label_elem))
            else:
                sampled_labels.append(label_elem)
        # sample ref images
        ref_images, n_shots = self.sample_ref_images_feature(sampled_labels)
        positive_maps = self.get_positive_map(gt_labels)
        # add info to results
        results['ref_images'] = ref_images
        results['ref_labels'] = sampled_labels
        results['n_shots'] = n_shots
        results['n_samples'] = len(set(gt_labels))
        
        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = gt_labels
        results['tokens_positive'] = positive_maps
        return results
import json
import re
import random
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

from mmcv.transforms import BaseTransform
from mmengine.structures import BaseDataElement
from mmdet.registry import TRANSFORMS
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
    def __init__(self, n_px: 256) -> None:
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
        for img in results['img']:
            inputs.append(self._clip_transform(img))
        results['inputs'] = inputs
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
        self.exemplar_dict = exemplar_dict
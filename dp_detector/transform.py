import os
import json
import torch
import numpy as np
import torchvision.transforms as transforms
import re

from PIL import Image
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SubLabel(BaseTransform):
    def __init__(self, label_map: str):
        self.label_map = json.load(open(label_map, "r"))

    @staticmethod
    def normalize(s: str) -> str:
        s, *_ = s.split("/", 1)
        s, *_ = s.split("(", 1)
        s, *_ = s.split("[", 1)
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = s.lower().strip()
        return s

    def transform(self, results: dict) -> dict:
        results["text"] = [
            self.label_map[self.normalize(label)] for label in results["text"]
        ]
        return results


@TRANSFORMS.register_module()
class LoadQwenFeature(BaseTransform):
    def __init__(
        self, qwen_feature_map: str, label_map: str, use_gt_name: bool = False
    ):
        self.label_map = json.load(open(label_map, "r"))
        self.qwen_feature_map = json.load(open(qwen_feature_map, "r"))
        self.use_gt_name = use_gt_name

    def transform(self, results: dict) -> dict:
        pth_list = []
        text_list = []
        for index in results["label_list"]:
            label = self.label_map[str(index)]
            pth, gt_name, name = torch.load(
                self.qwen_feature_map[label], map_location="cpu"
            )
            pth_list.append(pth.float()[3:])
            if self.use_gt_name:
                text_list.append(gt_name)
            else:
                text_list.append(name.replace("A photo of", "").strip())
        results["qwen_feature_list"] = pth_list
        results["text_list"] = text_list
        return results


@TRANSFORMS.register_module()
class LoadMMovodFinetuneFeature(BaseTransform):
    def __init__(self, mmovod_pseudo_list: str, label_map: str):
        data = torch.load(mmovod_pseudo_list, "cpu")
        self.pseudo_list = data["pseudo_list"]
        self.pth_list = data["pth_list"]
        self.label_map = json.load(open(label_map, "r"))

    def transform(self, results: dict) -> dict:
        results["text_list"] = [
            self.label_map[str(index)] for index in results["label_list"]
        ]
        results["qwen_feature_list"] = [
            self.pth_list[int(index)][3:].float() for index in results["label_list"]
        ]
        return results


@TRANSFORMS.register_module()
class LoadMMovodFeature(BaseTransform):
    def __init__(self, mmovod_pseudo_list: str):
        data = torch.load(mmovod_pseudo_list, "cpu")
        self.pseudo_list = data["pseudo_list"]
        self.pth_list = data["pth_list"]

    def transform(self, results: dict) -> dict:
        results["text"] = self.pseudo_list
        results["qwen_feature_list"] = [pth.float()[3:] for pth in self.pth_list]
        return results


@TRANSFORMS.register_module()
class LoadEnsembleMMovodFeature(BaseTransform):
    def __init__(self, mmovod_pseudo_list: str):
        data = torch.load(mmovod_pseudo_list, "cpu")
        self.pseudo_list = data["pseudo_list"]
        self.pth_list = data["pth_list"]

    def transform(self, results: dict) -> dict:
        results["real_text"] = results["text"]
        results["pseudo_text"] = self.pseudo_list
        results["qwen_feature_list"] = [pth.float()[3:] for pth in self.pth_list]
        return results


@TRANSFORMS.register_module()
class LoadEVAFeatures(BaseTransform):
    def __init__(self, dataset_name: str, data_dir: str, mode: str|None = None, clip_version: str = "eva-clip"):
        self.feature_level = ["globals", "blocks", "objects"]
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.clip_version = clip_version

    def transform(self, results: dict) -> dict:
        img_path = results["img_path"]
        if self.dataset_name == "llava-cap":
            if "/" in img_path:
                parts = img_path.split("/")
                pth_name = f"{parts[-2]}___{parts[-1]}.pth"
        elif self.dataset_name == "v3det":
            if "/" in img_path:
                parts = img_path.split("/")
                pth_name = f"{parts[-3]}___{parts[-2]}___{parts[-1]}.pth"
        else:
            pth_name = os.path.basename(img_path) + ".pth"
        for level in self.feature_level:
            if self.mode and level == "objects":
                feature_dir = os.path.join(
                    self.data_dir,
                    f"{level}_{self.dataset_name}_{self.clip_version}_{self.mode}",
                    "train",
                    pth_name,
                )
            else:
                feature_dir = os.path.join(
                    self.data_dir,
                    f"{level}_{self.dataset_name}_{self.clip_version}",
                    "train",
                    pth_name,
                )
            pth = torch.load(feature_dir, map_location="cpu")
            results[f"{level}_features"] = pth
        return results


@TRANSFORMS.register_module()
class LoadRefImages(BaseTransform):
    BASE_DIR = "data/grounding_data/imagenet-21k/images"

    def __init__(self, label_map: str, name2image: str):
        self.label_map = json.load(open(label_map, "r"))
        self.name2image = json.load(open(name2image, "r"))

    @staticmethod
    def normalize(s: str) -> str:
        s, *_ = s.split("/", 1)
        s, *_ = s.split("(", 1)
        s, *_ = s.split("[", 1)
        s = s.replace("_", " ")
        s = s.replace("-", " ")
        s = s.lower().strip()
        return s

    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        image = Image.open(image_path)
        image = image.resize((224, 224))

        # Convert grayscale images to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = torch.FloatTensor(np.array(image)).permute(2, 0, 1)
        image = transforms.Normalize(
            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]
        )(image)
        return image

    def transform(self, results: dict) -> dict:
        class_names = results["text"].split(". ")[:-1]  # filter the last empty string
        image_list = []
        for class_name in class_names:
            ref_images_path = self.name2image[self.normalize(class_name)].keys()
            ref_images = []
            for path in ref_images_path:
                real_path = os.path.join(
                    self.BASE_DIR, path.replace("data/imagenet-21k/", "")
                )
                ref_images.append(self.load_image(real_path))
            image_list.append(torch.stack(ref_images))
        results["ref_image_list"] = image_list
        return results

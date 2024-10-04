import argparse
import pathlib

import numpy as np
import todd
import torch
import torchvision.transforms.v2 as tf_v2
import tqdm
from einops.layers.torch import Rearrange
from PIL import Image
from todd.datasets import CLIP_MEAN, CLIP_STD
from todd.models.modules import CLIPViT
from todd.patches.py_ import decode_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    dataset: str = args.dataset

    transforms = tf_v2.Compose([
        Image.open,
        np.array,
        torch.from_numpy,
        Rearrange('h w c -> 1 c h w'),
        tf_v2.Resize(256, interpolation=tf_v2.InterpolationMode.BICUBIC),
        tf_v2.CenterCrop(256),
        tf_v2.ToDtype(torch.float32, True),
        tf_v2.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    model = CLIPViT(
        patch_size=32,
        patch_wh=(7, 7),
        out_features=512,
    )
    model.load_pretrained('pretrained/clip/ViT-B-32.pt')
    model.requires_grad_(False)
    model.eval()

    if todd.Store.cuda:
        model = model.cuda()

    work_dir = pathlib.Path('work_dirs/sample_images') / dataset
    embeddings: dict[str, torch.Tensor] = dict()

    for category in tqdm.tqdm(list(work_dir.iterdir())):
        tensor = torch.cat([
            transforms(category / f'{i}.png') for i in range(5)
        ])
        if todd.Store.cuda:
            tensor = tensor.cuda()
        tensor, _ = model(tensor, False)
        embeddings[decode_filename(category.name)] = tensor

    work_dir = pathlib.Path('work_dirs/sample_image_embeddings')
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, work_dir / f'{dataset}.pth')


if __name__ == '__main__':
    main()

import argparse
import pathlib

import todd
import torch
import torch.nn.functional as F
import tqdm
from clip import load_default
from PIL import Image
from todd.patches.py import decode_filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    dataset: str = args.dataset

    embeddings: dict[str, torch.Tensor] = dict()

    model, transforms = load_default(False)
    model.requires_grad_(False)

    work_dir = pathlib.Path('work_dirs/sample_images') / dataset
    for category in tqdm.tqdm(list(work_dir.iterdir())):
        images = [Image.open(category / f'{i}.png') for i in range(5)]
        tensor = torch.stack(list(map(transforms, images)))
        if todd.Store.cuda:
            tensor = tensor.cuda()
        tensor = model.encode_image(tensor)
        embeddings[decode_filename(category.name)] = F.normalize(tensor)

    work_dir = pathlib.Path('work_dirs/sample_image_embeddings')
    work_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, work_dir / f'{dataset}.pth')


if __name__ == '__main__':
    main()

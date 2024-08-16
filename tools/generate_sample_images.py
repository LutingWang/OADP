import argparse
import pathlib

import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from todd.loggers import logger
from todd.patches.torch import get_local_rank, get_rank, get_world_size

from oadp.categories import Categories

CACHE_DIR = 'pretrained/stable_diffusion/'
PRETRAINED = (
    CACHE_DIR
    + 'stable-diffusion-3-medium/sd3_medium_incl_clips_t5xxlfp8.safetensors'
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate sample images')
    parser.add_argument('dataset')
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    dataset: str = args.dataset

    dist.init_process_group()
    torch.cuda.set_device(get_local_rank() % torch.cuda.device_count())

    pipe: StableDiffusion3Pipeline = StableDiffusion3Pipeline.from_single_file(
        PRETRAINED,
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    pipe.to('cuda', torch.float)

    categories = Categories.get(dataset).all_[get_rank()::get_world_size()]
    for category in categories:
        logger.info("Generating sample images for %s", category)
        work_dir = pathlib.Path('work_dirs/sample_images') / dataset / category
        work_dir.mkdir(parents=True, exist_ok=True)
        output = pipe(category, num_images_per_prompt=5)
        for i, image in enumerate(output.images):
            image.save(work_dir / f'{i}.png')


if __name__ == '__main__':
    main()

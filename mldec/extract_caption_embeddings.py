import pathlib
from typing import Tuple

import clip
import todd
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

from .debug import debug


class CocoCaptions(torchvision.datasets.CocoCaptions):

    def __getitem__(self, index: int) -> Tuple[int, torch.IntTensor]:
        image_id = self.ids[index]
        target = super()._load_target(image_id)
        tokens = clip.tokenize(target, length=77)
        return image_id, tokens


@torch.no_grad()
def main() -> None:
    logger = todd.get_logger()
    debug.init()

    if debug.CPU:
        model, _ = clip.load('pretrained/clip/ViT-B-32.pt', 'cpu')
    else:
        model, _ = clip.load('pretrained/clip/ViT-B-32.pt')

    embeddings_root = pathlib.Path('data/coco/caption_embeddings/')
    train_embeddings_root = embeddings_root / 'train'
    train_embeddings_root.mkdir(parents=True, exist_ok=True)
    val_embeddings_root = embeddings_root / 'val'
    val_embeddings_root.mkdir(parents=True, exist_ok=True)

    logger.info('Train')
    dataset = CocoCaptions(
        root='data/coco/train2017',
        annFile='data/coco/annotations/captions_train2017.json',
    )
    for image_id, tokens in tqdm(dataset):
        if not debug.CPU:
            tokens = tokens.cuda()
        text = model.encode_text(tokens)
        text = F.normalize(text)
        torch.save(text, train_embeddings_root / f'{image_id:012d}.pth')

    logger.info('Val')
    dataset = CocoCaptions(
        root='data/coco/val2017',
        annFile='data/coco/annotations/captions_val2017.json',
    )
    for image_id, tokens in tqdm(dataset):
        if not debug.CPU:
            tokens = tokens.cuda()
        text = model.encode_text(tokens)
        text = F.normalize(text)
        torch.save(text, val_embeddings_root / f'{image_id:012d}.pth')


if __name__ == '__main__':
    main()

__all__ = [
    'ram_plus',
]

import todd.tasks.image_classification as ic
import torch
import torchvision.transforms.v2 as tf_v2
from todd.datasets import IMAGENET_MEAN, IMAGENET_STD
from todd.tasks.image_classification.models.ram import Categories
from torch import nn

from ..registries import OAKEModelRegistry


@OAKEModelRegistry.register_()
def ram_plus(
    expand_mask_size: int | None,
    adaptive: bool,
) -> tuple[nn.Module, tf_v2.Compose]:
    assert expand_mask_size is None
    assert not adaptive

    categories = Categories.load()
    model = ic.models.RAMplus(num_categories=len(categories))
    model.load_pretrained('pretrained/ram/ram_plus_swin_large_14m.pth')
    model.requires_grad_(False)
    model.eval()

    transforms = tf_v2.Compose([
        tf_v2.Resize((384, 384)),
        tf_v2.ToImage(),
        tf_v2.ToDtype(torch.float32, True),
        tf_v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return model, transforms

__all__ = [
    'Globals',
]

from typing import TYPE_CHECKING
import todd
import torch

if TYPE_CHECKING:
    from ..categories import Categories


class Globals(metaclass=todd.patches.py.NonInstantiableMeta):
    """Entry point for global variables.

    Not to be confused with the global distillation branch.
    """

    categories: 'Categories'
    training: bool
    visual_embeddings: torch.Tensor

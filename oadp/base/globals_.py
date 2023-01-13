__all__ = [
    'coco',
    'Globals',
]

import logging
from typing import Iterable

import todd


class Categories:

    def __init__(self, bases: Iterable[str], novels: Iterable[str]) -> None:
        self._bases = tuple(bases)
        self._novels = tuple(novels)

    @property
    def bases(self) -> tuple[str, ...]:
        return self._bases

    @property
    def novels(self) -> tuple[str, ...]:
        return self._novels

    @property
    def all_(self) -> tuple[str, ...]:
        return self._bases + self._novels

    @property
    def num_bases(self) -> int:
        return len(self._bases)

    @property
    def num_novels(self) -> int:
        return len(self._novels)

    @property
    def num_all(self) -> int:
        return len(self.all_)


coco = Categories(
    bases=(
        'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
        'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
        'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
        'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
        'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
        'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
        'toothbrush'
    ),
    novels=(
        'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
        'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
        'sink', 'scissors'
    ),
)


class Globals(metaclass=todd.NonInstantiableMeta):
    categories: Categories
    training: bool
    logger: logging.Logger = todd.logger

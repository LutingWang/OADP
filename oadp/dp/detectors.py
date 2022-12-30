__all__ = [
    'OADP',
]

from typing import NoReturn

import todd
from mmdet.models import DETECTORS, TwoStageDetector

from ..base import Store


@DETECTORS.register_module()
class OADP(TwoStageDetector):

    def __init__(
        self,
        *args,
        num_classes: int,
        num_base_classes: int,
        **kwargs,
    ) -> None:
        Store.NUM_CLASSES = num_classes
        Store.NUM_BASE_CLASSES = num_base_classes
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self) -> int:
        return Store.NUM_CLASSES

    def forward_train(self, *args, **kwargs) -> NoReturn:
        todd.Store.ITER += 1
        Store.TRAINING = True
        raise NotImplementedError

    def simple_test(self, *args, **kwargs):
        Store.TRAINING = False
        return super().simple_test(*args, **kwargs)

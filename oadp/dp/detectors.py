__all__ = [
    'OADP',
]

from typing import NoReturn

import todd
from mmdet.models import DETECTORS, TwoStageDetector


@DETECTORS.register_module()
class OADP(TwoStageDetector):

    def __init__(
        self,
        *args,
        num_classes: int,
        num_base_classes: int,
        **kwargs,
    ) -> None:
        todd.init_iter()
        todd.globals_.num_classes = num_classes
        todd.globals_.num_base_classes = num_base_classes
        super().__init__(*args, **kwargs)

    @property
    def num_classes(self) -> int:
        return todd.globals_.num_classes

    def forward_train(self, *args, **kwargs) -> NoReturn:
        todd.inc_iter()
        todd.globals_.training = True
        raise NotImplementedError

    def simple_test(self, *args, **kwargs):
        todd.globals_.training = False
        return super().simple_test(*args, **kwargs)

__all__ = [
    'OADP',
]

from typing import NoReturn

from mmdet.models import DETECTORS, TwoStageDetector

from ..base import Globals


@DETECTORS.register_module()
class OADP(TwoStageDetector):

    @property
    def num_classes(self) -> int:
        return Globals.categories.num_all

    def forward_train(self, *args, **kwargs) -> NoReturn:
        Globals.training = True
        raise NotImplementedError

    def simple_test(self, *args, **kwargs):
        Globals.training = False
        return super().simple_test(*args, **kwargs)

__all__ = [
    'debug',
]

from typing import Optional
import torch.nn as nn
import mmcv.cnn
import todd


class Debug(todd.base.BaseDebug):
    TRAIN_WITH_VAL_DATASET = todd.base.DebugMode()
    DRY_RUN = todd.base.DebugMode()

    def init_cpu(self, **kwargs) -> None:
        super().init_cpu(**kwargs)
        self.TRAIN_WITH_VAL_DATASET = True
        self.DRY_RUN = True
        self.SMALLER_BATCH_SIZE = True
        mmcv.cnn.NORM_LAYERS.register_module(name='SyncBN', force=True, module=nn.BatchNorm2d)

    def init(self, *, config: Optional[todd.base.Config] = None, **kwargs) -> None:
        super().init(**kwargs)
        if config is None:
            return
        if self.TRAIN_WITH_VAL_DATASET:
            val_dataset = {
                k: config.val.dataloader.dataset[k]
                for k in config.train.dataloader.dataset.keys() & config.val.dataloader.dataset.keys()
            }
            config.train.dataloader.dataset.update(val_dataset)
        if self.DRY_RUN:
            config.logger.interval = 1


debug = Debug()

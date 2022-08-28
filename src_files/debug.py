import torch
import todd


class Debug:
    CPU = todd.base.DebugMode()
    TRAIN_WITH_VAL_DATASET = todd.base.DebugMode()
    LESS_DATA = todd.base.DebugMode()
    SMALLER_BATCH_SIZE = todd.base.DebugMode()

    @classmethod
    def setup(cls, debug: bool, config: todd.base.Config) -> None:
        if torch.cuda.is_available():
            if debug:
                cls.LESS_DATA = True
            assert not cls.CPU
        else:
            cls.TRAIN_WITH_VAL_DATASET = True
            cls.LESS_DATA = True
            cls.CPU = True
            cls.SMALLER_BATCH_SIZE = True

        if cls.TRAIN_WITH_VAL_DATASET:
            config.train = config.val
        if cls.SMALLER_BATCH_SIZE:
            config.batch_size = 2

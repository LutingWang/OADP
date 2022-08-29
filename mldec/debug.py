import todd


class debug(todd.base.BaseDebug):
    TRAIN_WITH_VAL_DATASET = todd.base.DebugMode()
    LESS_DATA = todd.base.DebugMode()
    SMALLER_BATCH_SIZE = todd.base.DebugMode()

    def init_cpu(self, **kwargs) -> None:
        super().init_cpu(**kwargs)
        self.TRAIN_WITH_VAL_DATASET = True
        self.LESS_DATA = True
        self.SMALLER_BATCH_SIZE = True

    def init_cuda(self, *, debug: bool, **kwargs) -> None:
        super().init_cuda(**kwargs)
        if not debug:
            return
        self.TRAIN_WITH_VAL_DATASET = True
        self.LESS_DATA = True
        self.SMALLER_BATCH_SIZE = True

    def init_custom(self, *, config: todd.base.Config, **kwargs) -> None:
        super().init_custom(**kwargs)
        if self.TRAIN_WITH_VAL_DATASET:
            config.train = config.val
        if self.SMALLER_BATCH_SIZE:
            config.batch_size = 2


debug = debug()

import todd


class Debug(todd.base.BaseDebug):
    TRAIN_WITH_VAL_DATASET = todd.base.DebugMode()
    DRY_RUN = todd.base.DebugMode()
    SMALLER_BATCH_SIZE = todd.base.DebugMode()

    def init_cpu(self, **kwargs) -> None:
        super().init_cpu(**kwargs)
        self.TRAIN_WITH_VAL_DATASET = True
        self.DRY_RUN = True
        self.SMALLER_BATCH_SIZE = True

    def init(self, *, config: todd.base.Config, **kwargs) -> None:
        super().init(**kwargs)
        if self.TRAIN_WITH_VAL_DATASET:
            val_dataset = {
                k: config.val.dataloader.dataset[k]
                for k in config.train.dataloader.dataset.keys() & config.val.dataloader.dataset.keys()
            }
            config.train.dataloader.dataset.update(val_dataset)
        if self.SMALLER_BATCH_SIZE:
            config.train.batch_size = config.val.batch_size = 2


debug = Debug()

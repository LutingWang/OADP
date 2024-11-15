from mmengine.runner import Runner
from mmengine import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import exps.muti_label

if __name__ == "__main__":
    config = Config.fromfile("exps/muti_label/configs/ram_config.py")
    runner: Runner = RUNNERS.build(config)
    runner.val()
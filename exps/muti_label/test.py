import todd

from mmengine.runner import Runner
from mmengine import Config
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

import exps.muti_label

if __name__ == "__main__":
    config = Config.fromfile("/root/workspace/OADP/exps/muti_label/configs/clip_config.py")
    runner: Runner = RUNNERS.build(config)
    runner.val()
__all__ = [
    'DPRunner',
]

import pathlib
from todd import Config
from todd import Store
from mmengine.runner import Runner

from ..utils import Globals


class DPRunner(Runner):

    @classmethod
    def from_cfg(cls, config: Config, **kwargs) -> Runner:  # noqa: E501 pylint: disable=arguments-renamed
        config.update(kwargs)
        name = config.pop('name')
        work_dir = pathlib.Path('work_dirs') / name

        if Store.DRY_RUN:
            config.train_dataloader.num_workers = 0
            config.train_dataloader.persistent_workers = False

        if config.pop('autocast', False):
            config.optim_wrapper.type = 'AmpOptimWrapper'
            config.optim_wrapper.loss_scale = 'dynamic'

        if config.pop('auto_resume', False):
            config.resume = True
            config.load_from = None
        elif (load_from := config.pop('load_from', None)) is not None:
            config.resume = True
            config.load_from = load_from
        elif (
            load_model_from := config.pop('load_model_from', None)
        ) is not None:
            config.load_from = load_model_from

        config.work_dir = work_dir
        config.launcher = 'pytorch' if Store.cuda else 'none'

        if config.pop('visual', None) is not None:
            visualization_hook = config.default_hooks['visualization']
            visualization_hook['draw'] = True
            visualization_hook['test_out_dir'] = 'visual'

        return super().from_cfg(config)

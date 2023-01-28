import pathlib
import sys

from nni.experiment import Experiment

sys.path.insert(0, '')
from oadp.base import Store  # noqa: E402

experiment = Experiment('local')
experiment.config.experiment_name = 'dp_test'
experiment.config.trial_command = f'''
torchrun --nproc_per_node=1 -m oadp.dp.test_nni nni/dp_test \\
    configs/dp/oadp_ov_coco.py {Store.DUMP}
'''
experiment.config.trial_code_directory = pathlib.Path(__file__).parent.parent

experiment.config.search_space = dict(
    bbox_base_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    bbox_novel_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    bbox_base_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    bbox_novel_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    object_base_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    object_novel_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    object_base_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    object_novel_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    objectness_gamma=dict(
        _type='uniform',
        _value=[0, 1.0],
    ),
)

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1

experiment.run(8080)

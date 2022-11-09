import argparse
import pathlib

from nni.experiment import Experiment

parser = argparse.ArgumentParser(description='nni test multilabel fusion')
parser.add_argument('name')
parser.add_argument('trial')
args = parser.parse_args()

experiment = Experiment('local')

experiment.config.experiment_name = args.name

experiment.config.trial_command = f'''
python -m cafe.test_multilabel_fusion debug work_dirs/{args.name}/cafe_48_17.py work_dirs/{args.name}/{args.trial}
'''
experiment.config.trial_code_directory = pathlib.Path(__file__).parent.parent
experiment.config.experiment_working_directory = '/mnt/dolphinfs/ssd_pool/docker/user/hadoop-mtcv/dingzihan/openset/nni-experiments'

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
    image_base_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    image_novel_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    image_base_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    image_novel_gamma=dict(
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
experiment.config.trial_concurrency = 2

experiment.run(8080)

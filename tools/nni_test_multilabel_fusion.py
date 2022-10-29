import pathlib

from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'dcp_256_8_patch_16_128_0007_001'

experiment.config.trial_command = '''
python -m cafe.test_multilabel_fusion debug work_dirs/dcp_256_8_patch_16_128_0007_001/cafe_48_17.py work_dirs/dcp_256_8_patch_16_128_0007_001/debug1
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
experiment.config.trial_concurrency = 1

experiment.run(8080)

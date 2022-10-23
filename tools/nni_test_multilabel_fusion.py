import pathlib

from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'multilabel_post'

experiment.config.trial_command = '''
python -m cafe.test_multilabel_fusion debug work_dirs/patch_32_128/cafe_48_17.py work_dirs/patch_32_128/debug
'''
experiment.config.trial_code_directory = pathlib.Path(__file__).parent.parent

experiment.config.search_space = dict(
    base_ensemble_mask=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    novel_ensemble_mask=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    bbox_score_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.5],
    ),
    bbox_patch_scaler=dict(
        _type='uniform',
        _value=[0.5, 2.0],
    ),
    bbox_score_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    bbox_patch_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    bbox_objectness_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    image_score_scaler=dict(
        _type='uniform',
        _value=[0.2, 1.0],
    ),
    image_patch_scaler=dict(
        _type='uniform',
        _value=[0.5, 2.0],
    ),
    image_score_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
    image_patch_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    image_objectness_gamma=dict(
        _type='uniform',
        _value=[0.2, 0.8],
    ),
)

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 2

experiment.run(8080)

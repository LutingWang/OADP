import pathlib

from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'v1_multilabel'

experiment.config.trial_command = '''
python -m cafe.test_multilabel_fusion debug work_dirs/v1_multilabel/cafe_48_17.py work_dirs/v1_multilabel/debug
'''
experiment.config.trial_code_directory = pathlib.Path(__file__).parent.parent

experiment.config.search_space = dict(
    base_ensemble_mask=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    novel_ensemble_mask=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    bbox_score_scaler=dict(
        _type='uniform',
        _value=[0.2, 2.0],
    ),
    bbox_objectness_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    image_score_scaler=dict(
        _type='uniform',
        _value=[0.2, 2.0],
    ),
    image_objectness_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
)

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 2

experiment.run(8080)

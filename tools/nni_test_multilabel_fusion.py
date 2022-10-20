import pathlib

from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'debug'

experiment.config.trial_command = '''
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --master_port=0 \
    --nnodes=1 \
    -m mldec.test_vild test_vild work_dirs/oln/oln_test
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
        _value=[20, 200],
    ),
    bbox_objectness_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
    image_score_scaler=dict(
        _type='uniform',
        _value=[20, 200],
    ),
    image_objectness_gamma=dict(
        _type='uniform',
        _value=[0, 1],
    ),
)

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 4

experiment.run(5002)

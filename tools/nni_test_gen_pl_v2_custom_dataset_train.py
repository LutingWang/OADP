import pathlib

from nni.experiment import Experiment

experiment = Experiment('local')

experiment.config.experiment_name = 'gen_pl_v2_custom_dataset_train_test'

# experiment.config.trial_command = '''
# python \
#     -m mldec.test_gen_pl_v2_custom_dataset_train test_gen_pl_v2_custom_dataset_train_test --hotwater
# '''

experiment.config.trial_command = '''
torchrun \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    --master_port=0 \
    --nnodes=1 \
    -m mldec.test_gen_pl_v2_custom_dataset_train test_gen_pl_v2_custom_dataset_train_test 
'''
experiment.config.trial_code_directory = pathlib.Path(__file__).parent.parent

experiment.config.search_space = dict(
    top_KP=dict(
        _type='randint',
        _value=[60, 100],
    ),
    softmax_t=dict(
        _type='uniform',
        _value=[0.00001, 0.6],
    ),
    nms_iou_thres=dict(
        _type='uniform',
        _value=[0.5, 0.7],
    ),
    topK_clip_scores=dict(
        _type='randint',
        _value=[1, 2],
    ),
    bbox_objectness=dict(
        _type='choice',
        _value=[
            # dict(
            #     _name='add',
            #     clip_score_ratio=dict(
            #         _type='uniform',
            #         _value=[0, 1],
            #     ),
            #     obj_score_ratio=dict(
            #         _type='uniform',
            #         _value=[0, 1],
            #     ),
            # ),
            dict(
                _name='mul',
                clip_score_ratio=dict(
                    _type='uniform',
                    _value=[0.15, 0.9],
                ),
                obj_score_ratio=dict(
                    _type='uniform',
                    _value=[0.3, 1],
            ),)
        ],
    ),

)

experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

experiment.config.max_trial_number = 1000
experiment.config.trial_concurrency = 1

experiment.run(5014)

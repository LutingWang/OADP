_base_ = [
    '../datasets/pretrain.py',
    '../schedule/schedule_1x.py',
    '../models/oadp.py',
    '../default_runtime.py'
]


root = '/mnt/data1/wlt/workspace/OADP/'
work_dir = f'{root}work_dirs/pretrain'
set -e

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.10

pipenv run pip install /data/wlt/wheels/torch-2.4.1+cu121-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu121 torchvision==0.19.1+cu121

pipenv run pip install \
    nni \
    openmim \
    scikit-learn

pipenv run mim install mmcv
pipenv run mim install mmdet --no-deps  # mmdet requires mmcv<2.2.0
pipenv run pip install shapely terminaltables  # mmdet dependencies

pipenv run pip install \
    git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pipenv run make install_todd

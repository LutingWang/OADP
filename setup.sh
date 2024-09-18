set -e

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.3

git config --global --add safe.directory $(dirname $(realpath $0))

# pipenv run pip install -i https://download.pytorch.org/whl/cu118 torch==2.4.0+cu118 torchvision==0.19.0+cu118
pipenv run pip install /mnt/bn/wangluting/wheels/torch-2.4.0+cu118-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu118 torchvision==0.19.0+cu118
pipenv run pip install \
    nni \
    openmim \
    scikit-learn \

pipenv run mim install mmcv
pipenv run mim install mmdet --no-deps  # mmdet requires mmcv<2.2.0
pipenv run pip install shapely terminaltables  # mmdet dependencies

pipenv run pip install \
    git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021
pipenv run make install_todd

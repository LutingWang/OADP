GPUS=$1
PY_ARGS=${@:2}

torchrun --nproc_per_node=${GPUS} --master_port=${PORT:-29500} \
    $(dirname "$0")/train.py ${PY_ARGS}

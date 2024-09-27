set -e

for port in {5000..5100}; do
    if ! sudo netstat -tunlp | grep -w :${port} > /dev/null; then
        break
    fi
done

set -x

torchrun --nproc-per-node=$(nvidia-smi -L | wc -l) --master-port=${port} "$@"

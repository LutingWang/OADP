```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.extract_embeddings extract_embeddings configs/extract_extract_embeddings.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt configs/prompt.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt configs/prompt.py --load 3
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt configs/prompt_dump.py --load 3
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt_patched configs/prompt_patched.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt_patched configs/prompt_patched.py --load 3
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt_patched configs/prompt_dump.py --load 3
```

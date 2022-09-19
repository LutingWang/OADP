```shell
python -m mldec.extract_patch_features configs/extract_patch_features.py
```

```shell
python -m mldec.prompt train debug configs/prompt.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train debug configs/prompt.py
```

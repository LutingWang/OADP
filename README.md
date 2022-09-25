```shell
python -m mldec.extract_patch_features configs/extract_patch_features.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt configs/prompt.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt configs/prompt.py --load 3
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt configs/prompt.py --load 3
```

```shell
python -m mldec.image_refiner train debug configs/image_refiner.py --override .train.dataloader.workers:0 .val.dataloader.workers:0
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.image_refiner train image_refiner configs/image_refiner.py
```

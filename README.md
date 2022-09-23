```shell
python -m mldec.extract_patch_features configs/extract_patch_features.py
```

```shell
python -m mldec.prompt train debug configs/prompt.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train debug configs/prompt.py
torchrun --nproc_per_node=1 --master_port=5001 -m mldec.prompt val debug configs/prompt.py --load 1
python -m mldec.prompt dump debug configs/prompt.py --override train.workers:0 val.workers:0 --load 3
```

```shell
python -m mldec.image_refiner train debug configs/image_refiner.py --override .train.dataloader.workers:0 .val.dataloader.workers:0
```

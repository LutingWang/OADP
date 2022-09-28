```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.extract_embeddings extract_embeddings configs/mldec/extract_extract_embeddings.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt configs/mldec/prompt.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt configs/mldec/prompt.py --load 3
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt configs/mldec/prompt_dump.py --load 3
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt_patched configs/mldec/prompt_patched.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt_patched configs/mldec/prompt_patched.py --load 3
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt_patched configs/mldec/prompt_dump.py --load 3
```

```shell
python tools/train.py configs/cafe/faster_rcnn/classifier.py --cfg-options checkpoint_config.create_symlink=False evaluation.tmpdir=work_dirs/tmp123 --work-dir work_dirs/debug --odps GIT_COMMIT_ID:\'$(git rev-parse --short HEAD)\' TRAIN_WITH_VAL_DATASET:\'1\' DRY_RUN:\'1\'
sh tools/odps_train.sh custom_faster configs/cafe/faster_rcnn/classifier.py 8
```

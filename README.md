# README

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.extract_embeddings extract_embeddings configs/mldec/extract_embeddings.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.extract_mask_embeddings extract_mask_embeddings configs/mldec/extract_mask_embeddings.py
```

```shell
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt train prompt configs/mldec/prompt.py
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt val prompt configs/mldec/prompt.py --load 3
torchrun --nproc_per_node=1 --master_port=5000 -m mldec.prompt dump prompt configs/mldec/prompt_dump.py --load 3
```

```shell
python tools/train.py configs/cafe/faster_rcnn/classifier.py --cfg-options checkpoint_config.create_symlink=False evaluation.tmpdir=work_dirs/tmp123 --work-dir work_dirs/debug --odps GIT_COMMIT_ID:\'$(git rev-parse --short HEAD)\' TRAIN_WITH_VAL_DATASET:\'1\' DRY_RUN:\'1\'
DEBUG=1 sh tools/odps_train.sh debug configs/cafe/faster_rcnn/classifier.py 1
sh tools/odps_train.sh custom_faster configs/cafe/faster_rcnn/classifier.py 8
```

```shell
python tools/test.py configs/cafe/faster_rcnn/classifier.py work_dirs/debug/epoch_1.pth --eval bbox --cfg-options evaluation.tmpdir=work_dirs/tmp123 --odps GIT_COMMIT_ID:\'$(git rev-parse --short HEAD)\' TRAIN_WITH_VAL_DATASET:\'1\' DRY_RUN:\'1\'
```

```shell
python tools/test_multilabel.py configs/cafe/faster_rcnn/classifier.py --load work_dirs/debug/epoch_1.pth
DEBUG=1 sh tools/odps_test_multilabel.sh configs/cafe/faster_rcnn/classifier.py --load work_dirs/multilabel_faster_mlweight16/epoch_9.pth
sh tools/odps_test_multilabel.sh configs/cafe/faster_rcnn/classifier.py --load work_dirs/multilabel_faster_mlweight16/epoch_10.pth
```

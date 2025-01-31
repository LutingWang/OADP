auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/v3det/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::V3Det
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/v3det/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::V3Det
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/v3det/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::V3Det


auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/objects365v1/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::Objects365v1
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/objects365v1/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::Objects365v1
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/objects365v1/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::Objects365v1


auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/flickr/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::Fliker
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/flickr/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::Fliker
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/flickr/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::Fliker


auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/gqa/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::GQA
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/gqa/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::GQA
auto_torchrun -m oadp.oake.val /mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/oake/gqa/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::GQA



auto_torchrun tools/train.py configs/dp/dp_o365_goldg_v3det.py --work-dir=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtcv/weiziyu/109/OADP/work_dirs/dp_finetune


wget https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt




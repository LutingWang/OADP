auto_torchrun -m oadp.oake.val oake/v3det/clip_globals_cuda configs/oake/clip_globals_cuda.py --config-options dataset::V3Det
auto_torchrun -m oadp.oake.val oake/v3det/clip_blocks_cuda configs/oake/clip_blocks_cuda.py --config-options dataset::V3Det
auto_torchrun -m oadp.oake.val oake/v3det/clip_objects_cuda configs/oake/clip_objects_cuda.py --config-options dataset::V3Det
import sys

import todd
import torch

sys.path.insert(0, '')
from mldec import odps_init, k8s_init, debug
from mldec.extract_mask_embeddings import Trainer, parse_args


args = parse_args()
config = todd.base.Config.load(args.config)
if args.odps is not None:
    odps_init(args.odps)
if args.k8s is not None:
    k8s_init(args.k8s)
debug.init(config=config)
if args.override is not None:
    for k, v in args.override.items():
        todd.base.setattr_recur(config, k, v)

if not debug.CPU:
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(todd.base.get_local_rank())

todd.reproduction.init_seed(args.seed)

trainer = Trainer(name=args.name, config=config)
trainer.train()
trainer.run()

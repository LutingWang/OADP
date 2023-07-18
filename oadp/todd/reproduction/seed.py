__all__ = [
    'init_seed',
    'set_seed_temp',
]

import hashlib
import random
from contextlib import contextmanager
from typing import Generator

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from ..base import Store, get_world_size, logger


def randint(high: int = 2**30) -> int:
    seed = np.random.randint(high)
    if get_world_size() <= 1:
        return seed
    tensor = torch.tensor(seed, dtype=torch.int)
    dist.broadcast(tensor, src=0)
    return tensor.item() # type: ignore


def init_seed(seed: int) -> None:
    seed %= 2**30
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Store.CUDA:
        torch.cuda.manual_seed_all(seed)


@contextmanager
def set_seed_temp(
    seed=None,
    deterministic: bool = False,
    benchmark: bool = True,
) -> Generator[None, None, None]:
    if seed is None:
        seed = randint()
    elif isinstance(seed, int):
        pass
    else:
        if not isinstance(seed, bytes):
            seed = str(seed).encode()
        seed = hashlib.blake2b(seed, digest_size=4).hexdigest()
        seed = int(seed, 16)

    logger.info(f"Setting seed to {seed}.")

    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    if Store.CUDA:
        cuda_state = torch.cuda.get_rng_state()
        prev_deterministic = cudnn.deterministic
        prev_benchmark = cudnn.benchmark
        cudnn.deterministic = deterministic
        cudnn.benchmark = benchmark

    init_seed(seed)
    yield

    random.setstate(random_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)

    if Store.CUDA:
        torch.cuda.set_rng_state(cuda_state) # type: ignore
        cudnn.deterministic = prev_deterministic # type: ignore
        cudnn.benchmark = prev_benchmark # type: ignore

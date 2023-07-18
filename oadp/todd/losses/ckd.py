__all__ = [
    'ckd_loss',
    'CKDLoss',
]

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..base import BBoxesXYXY, get_rank, get_world_size
from .base import BaseLoss, LossRegistry


def ckd_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ignore = None,
    gamma: float = 0.07,
) -> torch.Tensor:
    """Wrapper of CKD loss.

    Refer to http://arxiv.org/abs/2108.07482.

    Args:
        pred: n x dim
            Normalized predictions.

        target: m x dim
            Normalized targets. First ``n`` targets correspond to ``pred``.

        ignore: (n, n)

    Returns:
        loss
    """
    if ignore is None:
        inds = torch.arange(
            pred.shape[0],
            dtype=torch.long,
            device=pred.device,
        )
        ignore = (inds, ) * 2
    similarity = target.mm(pred.t()) / gamma  # m x n
    similarity = similarity.exp()
    pos = torch.diag(similarity)  # n
    similarity = similarity.index_put(ignore, similarity.new_zeros(()))
    total = similarity.sum(0)  # n
    loss = pos / total  # n
    return -loss.log().mean()


class MemoryPool:

    def __init__(self, size: int = 10) -> None:
        self._memory: list[torch.Tensor] = []
        self._size = size
        self._rank = get_rank()
        self._world_size = get_world_size()

    def register(self, tensor: torch.Tensor) -> None:
        tensor = tensor.contiguous()
        if self._world_size > 1:
            tensor_list = [
                torch.zeros_like(tensor) for _ in range(self._world_size)
            ]
            dist.all_gather(tensor_list, tensor)
            tensor_list[0], tensor_list[self._rank] = \
                tensor_list[self._rank], tensor_list[0]
            tensor = torch.cat(tensor_list)
        else:
            tensor = tensor.detach()
        self._memory.insert(0, tensor)
        if len(self._memory) > self._size:
            self._memory.pop(-1)

    @property
    def memory(self) -> torch.Tensor:
        return torch.cat(self._memory)


@LossRegistry.register()
class CKDLoss(BaseLoss):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._memory_pool = MemoryPool()

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        bboxes = None,
    ) -> torch.Tensor:
        """Compute CKD loss.

        Refer to http://arxiv.org/abs/2108.07482.

        Args:
            preds: m x dim
            targets: m x dim
            bboxes: n x m x 4

        Returns:
            loss: 1
        """
        assert preds.shape == targets.shape, (preds.shape, targets.shape)
        preds = F.normalize(preds)
        targets = F.normalize(targets)
        self._memory_pool.register(targets)

        if bboxes is None:
            loss = ckd_loss(preds, self._memory_pool.memory)
        else:
            ignore_x, ignore_y, ind = [], [], 0
            for bbox in bboxes:
                bbox_ = BBoxesXYXY(bbox)
                ious = BBoxesXYXY.ious(bbox_, bbox_)
                x, y = torch.where(ious > 0.5)
                ignore_x.append(x + ind)
                ignore_y.append(y + ind)
                ind += len(bbox_)
            assert ind == preds.shape[0], (ind, preds.shape)
            ignore = (torch.cat(ignore_x), torch.cat(ignore_y))
            loss = ckd_loss(preds, self._memory_pool.memory, ignore)

        return loss

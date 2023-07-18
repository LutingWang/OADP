__all__ = [
    'CollectionTensor',
    'ListTensor',
]

import numbers
from typing import Callable, Generic, Iterable, Literal, TypeVar

import torch

T = TypeVar('T', torch.Tensor, list, tuple, dict)


class CollectionTensor(Generic[T]):

    @staticmethod
    def apply(feat: T, op: Callable[[torch.Tensor], torch.Tensor]) -> T:
        if isinstance(feat, torch.Tensor):
            return op(feat)
        if isinstance(feat, list):
            return [CollectionTensor.apply(f, op) for f in feat]
        if isinstance(feat, tuple):
            return tuple(  # type: ignore[return-value]
                CollectionTensor.apply(f, op) for f in feat
            )
        if isinstance(feat, dict):
            return {k: CollectionTensor.apply(v, op) for k, v in feat.items()}
        raise TypeError(f'Unknown type {type(feat)}.')

    @staticmethod
    def to(feat: T, device: torch.device) -> T:
        return CollectionTensor.apply(feat, lambda x: x.to(device))

    @staticmethod
    def cpu(feat: T) -> T:
        return CollectionTensor.apply(feat, lambda x: x.cpu())

    @staticmethod
    def cuda(feat: T) -> T:
        return CollectionTensor.apply(feat, lambda x: x.cuda())

    @staticmethod
    def allclose(
        feat1,
        feat2,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        if isinstance(feat1, torch.Tensor):
            assert isinstance(feat2, torch.Tensor)
            return torch.allclose(feat1, feat2)
        if isinstance(feat1, list) or isinstance(feat1, tuple):
            assert isinstance(feat2, list) or isinstance(feat2, tuple)
            assert len(feat1) == len(feat2)
            return all(  # yapf: disable
                CollectionTensor.allclose(f1, f2)
                for f1, f2 in zip(feat1, feat2)
            )
        if isinstance(feat1, dict):
            assert isinstance(feat2, dict) and feat1.keys() == feat2.keys()
            return all(
                CollectionTensor.allclose(feat1[k], feat2[k]) for k in feat1
            )
        if isinstance(feat1, numbers.Complex):
            assert isinstance(feat2, numbers.Complex)
            return abs(feat1 - feat2) <= atol + rtol * abs(feat2)
        raise TypeError(f'Unknown type {type(feat1)}.')

    @staticmethod
    def reduce(
        feat: T,
        tensor_op: Callable[[torch.Tensor], torch.Tensor],
        tensors_op,
    ):
        if isinstance(feat, torch.Tensor):
            return tensor_op(feat)
        if isinstance(feat, list) or isinstance(feat, tuple):
            return tensors_op([
                CollectionTensor.reduce(f, tensor_op, tensors_op) for f in feat
            ])
        if isinstance(feat, dict):
            return tensors_op([
                CollectionTensor.reduce(f, tensor_op, tensors_op)
                for f in feat.values()
            ])
        raise TypeError(f'Unknown type {type(feat)}.')

    @staticmethod
    def sum(feat: T):
        return CollectionTensor.reduce(feat, torch.sum, sum)


class ListTensor(CollectionTensor[T]):

    @staticmethod
    def stack(feat: T) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            return feat
        return torch.stack([ListTensor.stack(f) for f in feat])

    @staticmethod
    def shape(feat: T, depth: int = 0):
        if isinstance(feat, torch.Tensor):
            return feat.shape[max(depth, 0):]
        shapes = {ListTensor.shape(f, depth - 1) for f in feat}
        assert len(shapes) == 1
        shape = shapes.pop()
        if depth <= 0:
            shape = (len(feat), ) + shape
        return shape

    @staticmethod
    def new_empty(feat: T, *args, **kwargs) -> torch.Tensor:
        if isinstance(feat, torch.Tensor):
            return feat.new_empty(*args, **kwargs)
        return ListTensor.new_empty(feat[0], *args, **kwargs)

    @staticmethod
    def index(feat: T, pos: torch.Tensor) -> torch.Tensor:
        """Generalized ``feat[pos]``.

        Args:
            feat: d_0 x d_1 x ... x d_(n-1) x *
            pos: m x n

        Returns:
            indexed_feat: m x *
        """
        m, n = pos.shape
        if isinstance(feat, torch.Tensor):
            assert n <= feat.ndim
        if m == 0:
            shape = ListTensor.shape(feat, n)
            return ListTensor.new_empty(feat, 0, *shape)
        if n == 0:
            indexed_feat = ListTensor.stack(feat)
            return indexed_feat.unsqueeze(0).repeat(
                m,
                *[1] * indexed_feat.ndim,
            )

        pos = pos.long()
        if isinstance(feat, torch.Tensor):
            assert (pos >= 0).all()
            max_pos = pos.max(0).values
            feat_shape = pos.new_tensor(feat.shape)
            assert (max_pos < feat_shape[:n]).all(), \
                f'max_pos({max_pos}) larger than feat_shape({feat_shape}).'
            return feat[pos.split(1, 1)].squeeze(1) # type: ignore
        indices = []
        indexed_feats = []
        for i, f in enumerate(feat):
            index = pos[:, 0] == i
            if not index.any():
                continue
            index, = torch.where(index)
            indexed_feat = ListTensor.index(f, pos[index, 1:])
            indices.append(index)
            indexed_feats.append(indexed_feat)
        indexed_feat = torch.cat(indexed_feats)
        index = torch.cat(indices)
        assert index.shape == pos.shape[:1]
        indexed_feat[index] = indexed_feat.clone()
        return indexed_feat

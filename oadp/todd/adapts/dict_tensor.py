import itertools
import operator
from functools import reduce
from typing import Iterable, Iterator, Mapping, Sequence, overload, Tuple, List, Union

import einops
import torch

from .base import AdaptRegistry, BaseAdapt

__all__ = [
    'DictTensor',
    'Union_',
    'Intersect',
]

KeyType = Tuple[int, ...]


class DictTensor(Mapping):

    def __init__(
        self,
        keys: Union[KeyType , List[KeyType]],
        values: torch.Tensor,
    ) -> None:
        if not isinstance(keys, list):
            keys = [keys]
        self._keys = {key: i for i, key in enumerate(keys)}
        self._values = values

    @overload
    def __getitem__(self, keys: KeyType) -> torch.Tensor:
        ...

    @overload
    def __getitem__(self, keys: List[KeyType]) -> torch.Tensor:
        ...

    def __getitem__(self, keys):
        inds = (
            self._keys[keys] if not isinstance(keys, list) else
            [self._keys[key] for key in keys]
        )
        return self._values[inds]

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[KeyType]:
        return iter(self._keys)

    def __contains__(self, keys: object) -> bool:
        if not isinstance(keys, list):
            return keys in self._keys
        return all(key in self._keys for key in keys)

    @classmethod
    def from_tensor(
        cls,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> 'DictTensor':
        return DictTensor(list(map(tuple, keys.int().tolist())), values)

    @staticmethod
    def union(
        dict_tensors: Sequence['DictTensor'],
    ) -> Tuple[List['DictTensor'], 'DictTensor']:
        keys = list(set(itertools.chain(*dict_tensors)))
        n = len(keys)
        s = len(dict_tensors)

        values = dict_tensors[0]._values.new_zeros((n, s))
        mask = DictTensor(keys, values)

        union_dict_tensors = []
        for i, dict_tensor in enumerate(dict_tensors):
            inds = [mask._keys[key] for key in dict_tensor._keys]
            mask._values[inds, i] = 1

            shape = (n, ) + dict_tensor._values.shape[1:]
            values = dict_tensor._values.new_zeros(shape)
            values[inds] = dict_tensor._values
            union_dict_tensor = DictTensor(keys, values)

            union_dict_tensors.append(union_dict_tensor)
        return union_dict_tensors, mask

    @staticmethod
    def intersect(dict_tensors: Iterable['DictTensor']) -> List['DictTensor']:
        keys: list[KeyType] = list(
            reduce(operator.and_, map(set, dict_tensors)),
        )
        dict_tensors = [
            DictTensor(keys, dict_tensor[keys]) for dict_tensor in dict_tensors
        ]
        return dict_tensors


@AdaptRegistry.register(keys=('Union', ))
class Union_(BaseAdapt):

    def forward(
        self,
        feats: List[torch.Tensor],
        ids: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Match `feats` according to their `poses`.

        Align the `feats` coming from different sources to have same
        `matched_pos` and stack them togethor. For positions where some
        of `feats` do not show up, an all-zero tensor is added as
        default. A 2D `mask` is returned to indicate the type of a
        matched feature, where `1` corresponds to features coming from
        `feats` and `0` for added default all-zero tensors.

        Args:
            feats: [n_s x d_1 x d_2 x ... x d_m]
                Features from `s` different sources, each source can
                have different `n_s`.
            ids: [n_s x l]
                Positions of each feature.

        Returns:
            union_feats: s x n x d_1 x d_2 x ... x d_m
            union_ids: n x l
            union_mask: s x n
        """
        dict_tensors = [
            DictTensor.from_tensor(id_, feat) for id_, feat in zip(ids, feats)
        ]
        dict_tensors, mask = DictTensor.union(dict_tensors)
        union_feats = torch.stack([
            dict_tensor._values for dict_tensor in dict_tensors
        ])
        union_ids = union_feats.new_tensor(mask._keys)
        union_mask = einops.rearrange(mask._values, 'n s -> s n')
        return union_feats, union_ids, union_mask


@AdaptRegistry.register()
class Intersect(BaseAdapt):

    def forward(
        self,
        feats: List[torch.Tensor],
        ids: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """Match positions that show up both in `pred_poses` and
        `target_poses`.

        Args:
            feats: [n_s x d_1 x d_2 x ... x d_m]
                Features from `s` different sources, each source can
                have different `n_s`.
            ids: [n_s x l]
                Positions of each feature.

        Returns:
            intersect_feats: [n x d_1 x d_2 x ... x d_m]
        """
        dict_tensors = [
            DictTensor.from_tensor(id_, feat) for id_, feat in zip(ids, feats)
        ]
        dict_tensors = DictTensor.intersect(dict_tensors)
        intersect_feats = [dict_tensor._values for dict_tensor in dict_tensors]
        return intersect_feats

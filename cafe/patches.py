_base_ = [
    'one_hot',
]

from typing import Sequence, TypeGuard, Union
import torch


def _is_tensor_sequence(data) -> TypeGuard[Sequence[torch.Tensor]]:
    if not isinstance(data, Sequence):
        return False
    return all(isinstance(datum, torch.Tensor) for datum in data)


def one_hot(labels: Union[torch.Tensor, Sequence[torch.Tensor]], num_classes: int) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        assert labels.ndim == 2
        y = labels.new_zeros(
            labels.shape[0],
            num_classes,
            dtype=bool,
        )
    elif _is_tensor_sequence(labels):
        assert all(label.ndim == 1 for label in labels)
        y = labels[0].new_zeros(
            len(labels),
            num_classes,
            dtype=bool,
        )
    else:
        raise TypeError(f"Unsupported type {type(labels)}")

    for i, label in enumerate(labels):
        y[i, label] = True
    return y

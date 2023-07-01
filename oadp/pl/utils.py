import todd
import torch
import torch.distributed


def all_gather(tensors_, shape) -> list[torch.Tensor]:
    tensor = torch.cat(tensors_)
    tensors = []
    for _ in range(todd.base.get_world_size()):
        if len(tensor.shape) == 2:
            tensors.append(
                torch.zeros([int(shape), tensor.shape[1]],
                            device=tensor.device,
                            dtype=tensor.dtype)
            )
        else:
            tensors.append(
                torch.zeros([int(shape)],
                            device=tensor.device,
                            dtype=tensor.dtype)
            )
    if len(tensor.shape) == 2:
        fake_tensor = torch.zeros([
            int(shape) - tensor.shape[0], tensor.shape[1]
        ],
                                  device=tensor.device,
                                  dtype=tensor.dtype)
    else:
        fake_tensor = torch.zeros([int(shape) - tensor.shape[0]],
                                  device=tensor.device,
                                  dtype=tensor.dtype)
    tensor = torch.cat((tensor, fake_tensor))
    torch.distributed.all_gather(tensors, tensor)
    return tensors


def all_gather_shape(tensors_: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
    tensor = torch.cat(tensors_)
    tensors = [
        torch.zeros(1, device=tensor.device)[0]
        for _ in range(todd.base.get_world_size())
    ]
    torch.distributed.all_gather(
        tensors,
        torch.tensor(
            tensor.shape[0], device=tensor.device, dtype=tensors[0].dtype
        )
    )
    return tensors

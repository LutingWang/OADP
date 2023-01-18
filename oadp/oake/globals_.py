import argparse
import pathlib
from typing import NamedTuple

import clip
import clip.model
import todd
import torch
import torch.cuda
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision


class Batch(NamedTuple):
    id_: int
    image: torch.Tensor


class CocoClassification(torchvision.datasets.CocoDetection):

    def __getitem__(self, index: int) -> Batch:
        id_ = self.ids[index]
        image = self._load_image(id_)
        image, _ = self.transforms(image, None)
        return Batch(id_, image)


class Validator(todd.utils.Validator):
    _model: clip.model.CLIP

    def __init__(self, *args, output_dir: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        if todd.Store.DRY_RUN:
            config.num_workers = 0
        config.dataset = CocoClassification(**config.dataset)
        if todd.Store.CUDA:
            config.sampler = torch.utils.data.distributed.DistributedSampler(
                config.dataset,
                shuffle=False,
            )
        return torch.utils.data.DataLoader(batch_size=None, **config)

    def _run_iter(self, batch: Batch, memo: todd.utils.Memo) -> torch.Tensor:
        super()._run_iter(batch, memo)
        image = batch.image.unsqueeze(0)
        if todd.Store.CUDA:
            image = image.cuda()
        image = self._model.encode_image(image)
        image = F.normalize(image)
        memo['result'] = dict(image=image.half())
        return torch.tensor(0.0)

    def _after_run_iter(self, batch: Batch, memo: todd.utils.Memo) -> None:
        super()._after_run_iter(batch, memo)
        output = self._output_dir / f'{batch.id_:012d}.pth'
        torch.save(memo['result'], output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('--override', action=todd.DictAction)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)

    if todd.Store.CUDA:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.base.get_local_rank())

    model, preprocess = clip.load_default(False)

    train = config.pop('train')
    val = config.pop('val')
    train.dataloader.dataset.transform = preprocess
    val.dataloader.dataset.transform = preprocess

    Validator(
        args.name,
        model,
        state_dict=todd.Config(),
        load_state_dict=todd.Config(),
        **val,
        **config,
    ).run()

    Validator(
        args.name,
        model,
        state_dict=todd.Config(),
        load_state_dict=todd.Config(),
        **train,
        **config,
    ).run()


if __name__ == '__main__':
    main()

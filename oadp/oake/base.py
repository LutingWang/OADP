import argparse
import pathlib
from abc import ABC, abstractmethod
from typing import Generic, Protocol, TypeVar

import clip
import clip.model
import PIL.Image
import todd
import torch
import torch.distributed
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms


class Batch(Protocol):

    @property
    def output(self) -> pathlib.Path:
        ...


T = TypeVar('T', bound=Batch)


class BaseDataset(torchvision.datasets.CocoDetection, ABC, Generic[T]):

    def __init__(
        self,
        *args,
        auto_fix: bool = False,
        output_dir: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._auto_fix = auto_fix
        self._output_dir = pathlib.Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, index: int) -> T | None:
        id_ = self.ids[index]
        output = self._output_dir / f'{id_:012d}.pth'
        if output.exists():
            if not self._auto_fix:
                return None
            try:
                torch.load(output, 'cpu')
                return None
            except Exception:
                todd.logger.info(f"Fixing {output}")
        image = self._load_image(id_)
        return self._preprocess(id_, output, image)

    @abstractmethod
    def _preprocess(
        self,
        id_: int,
        output: pathlib.Path,
        image: PIL.Image.Image,
    ) -> T:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('--override', action=todd.base.DictAction)
    args = parser.parse_args()
    return args


class BaseValidator(todd.utils.Validator, Generic[T]):
    _model: clip.model.CLIP

    def _build_dataloader(
        self,
        config: todd.Config,
    ) -> torch.utils.data.DataLoader:
        if todd.Store.DRY_RUN:
            config.num_workers = 0
        if todd.Store.CUDA:
            config.sampler = torch.utils.data.distributed.DistributedSampler(
                config.dataset,
                shuffle=False,
            )
        return torch.utils.data.DataLoader(batch_size=None, **config)

    @classmethod
    @abstractmethod
    def _build_model(cls) -> tuple[clip.model.CLIP, transforms.Compose]:
        pass

    def _control_run_iter(
        self,
        batch: T | None,
        memo: todd.utils.Memo,
    ) -> todd.utils.Control | None:
        control = super()._control_run_iter(batch, memo)
        if control is None and batch is None:
            return todd.utils.Control.CONTINUE
        return control

    def _run_iter(
        self,
        batch: T,
        memo: todd.utils.Memo,
    ) -> torch.Tensor:
        super()._run_iter(batch, memo)
        torch.save(memo['result'], batch.output)
        return torch.tensor(0.0)

    @classmethod
    def main(cls) -> None:
        args = parse_args()
        config: todd.Config = args.config
        if args.override is not None:
            config.override(args.override)

        if todd.Store.CUDA:
            torch.distributed.init_process_group(backend='nccl')
            torch.cuda.set_device(
                todd.get_local_rank() % torch.cuda.device_count()
            )

        model, preprocess = cls._build_model()

        train = config.pop('train')
        val = config.pop('val')
        train.dataloader.dataset.transform = preprocess
        val.dataloader.dataset.transform = preprocess

        # FIXME: in todd, this creates two log files
        cls(
            args.name,
            model,
            state_dict=todd.Config(),
            load_state_dict=todd.Config(),
            **val,
            **config,
        ).run()

        cls(
            args.name,
            model,
            state_dict=todd.Config(),
            load_state_dict=todd.Config(),
            **train,
            **config,
        ).run()

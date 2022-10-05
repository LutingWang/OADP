import argparse
import enum
import math
import os
import pathlib
import pickle
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, cast

import PIL.Image
import clip
import clip.model
import einops
import todd
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms

from .debug import debug
from .todd import BaseRunner, TrainerMixin
from .utils import odps_init, k8s_init

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC


class Batch(NamedTuple):
    image_id: int
    patches: torch.Tensor
    bboxes: torch.Tensor
    objectness: torch.Tensor
    masks: torch.Tensor


class ExpandMode(enum.IntEnum):
    RECTANGLE = enum.auto()
    LONGEST_EDGE = enum.auto()
    CONSTANT = enum.auto()
    ADAPTIVE = enum.auto()


class CocoClassification(torchvision.datasets.CocoDetection):

    def __init__(
        self,
        root: str,
        ann_file: str,
        proposal_file: str,
        mask_size: int,
        expand_mode: str,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )

        with open(proposal_file,'rb') as f:
            self.proposals = torch.tensor(
                pickle.load(f),
                dtype=torch.float,
            )

        self._mask_size = mask_size
        self._expand_mode = ExpandMode[expand_mode.upper()]

        self._transform = transforms.Compose([
            transforms.Resize(224, interpolation=BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def _crop(
        self,
        image: PIL.Image.Image,
        bboxes: todd.base.BBoxesXYXY,
        **kwargs,
    )-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: image to be cropped
            bboxes: n x 4

        Returns:
            patches: n x 3 x 224 x 224, cropped bboxes
            masks: n x 197, processed transformer masks
        """

        if self._expand_mode == ExpandMode.LONGEST_EDGE:
            max_length = einops.rearrange(
                torch.max(bboxes.wh, 1).values,
                'n -> n 1',
            )
        elif self._expand_mode == ExpandMode.CONSTANT:
            max_length = torch.full((len(bboxes), 1), 224)
        elif self._expand_mode == ExpandMode.ADAPTIVE:
            scale_ratio = kwargs.get('scale_ratio', 8)
            max_length = einops.rearrange(
                torch.sqrt(bboxes.areas * scale_ratio),
                'n -> n 1',
            )
        else:
            assert ValueError(self._expand_mode)

        lt = bboxes.center - max_length / 2
        rb = bboxes.center + max_length / 2

        image_size = torch.tensor(image.size)
        offset = torch.zeros(len(bboxes), 2)
        offset = torch.where(lt >= 0, offset, lt)
        offset = torch.where(rb <= image_size, offset, rb - image_size)
        offset = torch.where(max_length <= min(image_size), offset, torch.tensor(0.0))

        lt -= offset
        rb -= offset
        expanded_bboxes = torch.cat([lt, rb], -1).round().int()

        mask_lt = bboxes.lt - lt
        mask_rb = bboxes.rb - lt
        mask_bboxes = torch.cat([mask_lt, mask_rb], -1)

        patches = []
        masks = []
        for mask_bbox, expanded_bbox in zip(mask_bboxes.tolist(), expanded_bboxes.tolist()):
            patch = image.crop(expanded_bbox)
            patches.append(self._transform(patch))

            x = torch.arange(expanded_bbox[2] - expanded_bbox[0])
            w_mask: torch.Tensor = einops.rearrange((mask_bbox[0] <= x) & (x <= mask_bbox[2]), 'w -> 1 w')
            y = torch.arange(expanded_bbox[3] - expanded_bbox[1])
            h_mask: torch.Tensor = einops.rearrange((mask_bbox[1] <= y) & (y <= mask_bbox[3]), 'h -> h 1')
            mask: torch.Tensor = w_mask & h_mask
            mask = einops.rearrange(~mask, 'h w -> 1 1 h w')  # 0 for the object and 1 for others
            mask = F.interpolate(mask.float(), size=(self._mask_size, self._mask_size), mode='nearest')
            mask = torch.cat([mask.flatten(), torch.zeros(1)]) * -100
            masks.append(mask.half())

        return torch.stack(patches), torch.stack(masks)

    def __getitem__(self, index: int) -> Batch:
        image_id = self.ids[index]
        image = self._load_image(image_id)
        proposals = self.proposals[index]
        bboxes = proposals[:, :4]
        objectness = proposals[:,-1]
        bboxes[:, 2:] += bboxes[:, :2]
        patches, masks = self._crop(image, todd.base.BBoxesXYXY(bboxes))
        return Batch(image_id, patches, bboxes, objectness, masks)

    @staticmethod
    def collate(batch: List[Batch]) -> Batch:
        assert len(batch) == 1
        return batch[0]


class Model(todd.reproduction.FrozenMixin, todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        patch_size: int,
        upsample: int,
        **kwargs,
    ) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        visual = clip_model.visual

        conv = visual.conv1
        conv.stride = (patch_size // upsample,) * 2
        conv.padding = ((patch_size - 1) // 2,) * 2
        self._conv = conv

        pe = visual.positional_embedding
        self._spatial_size, self._pe = self.upsample_pe(pe, upsample)

        self._ce = visual.class_embedding
        self._ln_pre = visual.ln_pre
        self._transformer = visual.transformer
        self._ln_post = visual.ln_post
        self._proj = visual.proj

        eval_names = ['_conv', '_ln_pre', '_transformer', '_ln_post']
        no_grad_names = eval_names + ['_pe', '_ce', '_proj']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=no_grad_names),
            eval_config=dict(names=eval_names),
        )

    @staticmethod
    def upsample_pe(pe: torch.Tensor, upsample: int) -> Tuple[int, torch.Tensor]:
        cls_pe = pe[:1, :]
        spatial_pe = pe[1:, :]
        s2 = spatial_pe.shape[0]
        spatial_size = int(s2 **0.5)
        assert spatial_size ** 2 == s2
        spatial_pe = einops.rearrange(spatial_pe, '(h w) c -> 1 c h w', h=spatial_size, w=spatial_size)
        spatial_size *= upsample
        spatial_pe = F.interpolate(spatial_pe, size=spatial_size, mode='bilinear')
        spatial_pe = einops.rearrange(spatial_pe, '1 c h w -> (h w) c')
        pe = torch.cat([cls_pe, spatial_pe])
        pe = nn.parameter.Parameter(pe.half())
        return spatial_size, pe

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = self._conv(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self._ce.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self._pe.to(x.dtype)
        x = self._ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        y = x[[0]]

        resblocks = cast(Sequence[clip.model.ResidualAttentionBlock], self._transformer.resblocks)
        attn_mask = einops.repeat(attn_mask, 'b v -> (b h) 1 v', h=resblocks[0].attn.num_heads)
        for resblock in resblocks:
            source = resblock.ln_1(torch.cat([x[1:], y]))
            y = y + resblock.attn(
                source[[-1]],
                source,
                source,
                need_weights=False,
                attn_mask=attn_mask,
            )[0]
            y = y + resblock.mlp(resblock.ln_2(y))
            x = resblock(x)

        y = y.permute(1, 0, 2)

        y = self._ln_post(y)

        if self._proj is not None:
            y = y @ self._proj

        y = einops.rearrange(y, 'b 1 c -> b c')
        y = F.normalize(y)
        return y


class Runner(BaseRunner):

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(**config.dataset)
        sampler = (
            None if debug.CPU else
            torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=False,
            )
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=dataset.collate,
        )
        return dataset, sampler, dataloader

    def _build_model(
        self,
        *args,
        config: Optional[todd.base.Config],
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        config = config.copy()
        pretrained = config.pop('pretrained')
        assert "ViT-B-32" in pretrained
        if debug.CPU:
            clip_model, _ = clip.load(pretrained, 'cpu')
        else:
            clip_model, _ = clip.load(pretrained)
        clip_model.requires_grad_(False)
        model = Model(clip_model=clip_model, **config)
        if not debug.CPU:
            model = model.cuda()
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        embeddings_root = pathlib.Path(self._config.embeddings_root) / 'val'
        embeddings_root.mkdir(parents=True, exist_ok=True)
        memo['embeddings_root'] = embeddings_root
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        patches = batch.patches
        masks = batch.masks
        if debug.DRY_RUN:
            patches = patches[:5]
            masks = masks[:5]
        if not debug.CPU:
            patches = patches.half().cuda()
            masks = masks.half().cuda()
        patches_list = []
        for i in range(math.ceil(patches.shape[0] / self._config.mini_batch_size)):
            patches_list.append(
                self._model(
                    patches[i * self._config.mini_batch_size:(i + 1) * self._config.mini_batch_size],
                    masks[i * self._config.mini_batch_size:(i + 1) * self._config.mini_batch_size],
                ),
            )
        memo['result']=dict(
            bboxes=batch.bboxes.half(),
            patches=torch.cat(patches_list).half(),
            objectness = batch.objectness.half()
        )

    def _after_run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        torch.save(
            memo.pop('result'),
            memo['embeddings_root'] / f'{batch.image_id:012d}.pth',
        )
        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(self, *args, **kwargs) -> Tuple[
        CocoClassification,
        None,
        torch.utils.data.DataLoader,
    ]:
        return self._build_dataloader(*args, **kwargs)

    def _build_train_fixtures(self, *args, **kwargs) -> None:
        pass

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = self._before_run(*args, **kwargs)
        embeddings_root = memo['embeddings_root'].parent / 'train'
        embeddings_root.mkdir(parents=True, exist_ok=True)
        memo['train_embeddings_root'] = embeddings_root
        return memo

    @torch.no_grad()
    def _train_iter(self, *args, **kwargs) -> None:
        return self._run_iter(*args, **kwargs)

    def _after_train_iter(
        self,
        *args,
        epoch: int,
        i: int,
        batch: Batch,
        memo: Dict[str, Any],
        log: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        torch.save(
            memo.pop('result'),
            memo['train_embeddings_root'] / f'{batch.image_id:012d}.pth',
        )
        if log and todd.base.get_rank() == 0:
            self._logger.info(
                f'Train Step [{i}/{len(self._train_dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.base.DictAction)
    parser.add_argument('--override', action=todd.base.DictAction)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--k8s', action=todd.base.DictAction)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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
        if args.local_rank is not None:
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            torch.cuda.set_device(args.local_rank)
        else:
            torch.cuda.set_device(todd.base.get_local_rank())

    todd.reproduction.init_seed(args.seed)

    trainer = Trainer(name=args.name, config=config)
    trainer.train()
    trainer.run()

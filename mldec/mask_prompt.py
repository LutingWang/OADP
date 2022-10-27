import argparse
import enum
import os
import pathlib
from typing import Any, Dict, Iterator, List, MutableMapping, NamedTuple, Optional, Sequence, Tuple, cast

import PIL.Image
import sklearn.metrics
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

import mldec

from .debug import debug
from .todd import BaseRunner, TrainerMixin
from .utils import all_gather, odps_init, k8s_init

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC


class Batch(NamedTuple):
    patches: torch.Tensor
    masks: torch.Tensor
    labels: torch.Tensor
    class_embeddings: torch.Tensor
    class_lengths: torch.Tensor


class ExpandMode(enum.IntEnum):
    RECTANGLE = enum.auto()
    LONGEST_EDGE = enum.auto()
    CONSTANT = enum.auto()
    ADAPTIVE = enum.auto()


class CocoClassification(torchvision.datasets.CocoDetection):

    def __init__(
        self,
        *,
        root: str,
        ann_file: str,
        split: str,
        clip_model: clip.model.CLIP,
        mask_size: int,
        expand_mode: str,
    ) -> None:
        super().__init__(
            root=root,
            annFile=ann_file,
        )

        classnames = getattr(mldec, split)
        self._classnames: List[str] = []
        self._cat2label: Dict[str, int] = dict()
        for cat in self.coco.cats.values():
            if cat['name'] in classnames:
                self._classnames.append(cat['name'])
                self._cat2label[cat['id']] = len(self._cat2label)

        self.ids = list(filter(self._load_target, self.ids))  # filter empty

        class_tokens = clip.tokenize([
            name.replace('_', ' ') + '.'
            for name in self._classnames
        ])
        with torch.no_grad():
            self._class_embeddings = clip_model.token_embedding(class_tokens)
            self._class_lengths = class_tokens.argmax(dim=-1)

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

    @property
    def classnames(self) -> Tuple[str]:
        return tuple(self._classnames)

    @property
    def num_classes(self) -> int:
        return len(self._classnames)

    def _load_target(self, *args, **kwargs) -> List[Any]:
        target = super()._load_target(*args, **kwargs)
        return [anno for anno in target if anno['category_id'] in self._cat2label]

    def _crop(
        self,
        image: PIL.Image.Image,
        bboxes: todd.BBoxesXYXY,
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
                torch.sqrt(bboxes.area * scale_ratio),
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)
        bboxes = todd.BBoxesXYXY(todd.BBoxesXYWH([anno['bbox'] for anno in target]))

        patches, masks = self._crop(image, bboxes)
        bbox_labels = torch.tensor([self._cat2label[anno['category_id']] for anno in target])
        return patches, masks, bbox_labels

    def collate(self, batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Batch:
        batch = map(torch.cat, zip(*batch))
        return Batch(*batch, self._class_embeddings, self._class_lengths)


class ImageEncoder(todd.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        patch_size: int,
        upsample: int,
        **kwargs,
    ) -> None:
        todd.Module.__init__(self, *args, **kwargs)
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

    def forward(self, batch: Batch) -> torch.Tensor:
        x: torch.Tensor = batch.patches
        x = self._conv(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self._ce.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self._pe.to(x.dtype)
        x = self._ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        y = x[[0]]

        resblocks = cast(Sequence[clip.model.ResidualAttentionBlock], self._transformer.resblocks)
        attn_mask = einops.repeat(batch.masks, 'b v -> (b h) 1 v', h=resblocks[0].attn.num_heads)
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


class TextPrompt(todd.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        prompt: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompt_tokens = clip.tokenize([prompt])[0, 1:-1]
        with torch.no_grad():
            prompt_embedding = clip_model.token_embedding(prompt_tokens)
        self._prompt_embedding = nn.Parameter(prompt_embedding)

    def forward(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embeddings = einops.repeat(
            self._prompt_embedding,
            'l d -> n l d',
            n=batch.class_embeddings.shape[0],
        )
        x = torch.cat(
            [
                batch.class_embeddings[:, :1],
                prompt_embeddings,
                batch.class_embeddings[:, 1:],
            ],
            dim=1,
        )
        l = batch.class_lengths + self._prompt_embedding.shape[0]
        return x, l


class TextEncoder(todd.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transformer = clip_model.transformer
        self._pe = clip_model.positional_embedding
        self._ln = clip_model.ln_final
        self._proj = clip_model.text_projection

    def forward(self, batch: Batch, prompt: TextPrompt) -> torch.Tensor:
        x, l = prompt(batch)
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self._proj
        return F.normalize(x)


class Model(todd.reproduction.FrozenMixin, todd.Module):

    def _is_frozen(self, key: str) -> bool:
        return any(name in key for name in self._frozen_names)

    @staticmethod
    def _state_dict_hook(
        self: 'Model',
        destination: MutableMapping[str, Any],
        prefix,
        local_metadata,
    ) -> None:
        for k in list(filter(self._is_frozen, destination.keys())):
            destination.pop(k)

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        image_encoder: todd.Config,
        text_prompt: todd.Config,
        text_encoder: todd.Config,
        **kwargs,
    ) -> None:
        todd.Module.__init__(self, *args, **kwargs)
        self._image_encoder = ImageEncoder(
            clip_model=clip_model,
            **image_encoder,
        )
        self._text_prompt = TextPrompt(
            clip_model=clip_model,
            **text_prompt,
        )
        self._text_encoder = TextEncoder(
            clip_model=clip_model,
            **text_encoder,
        )
        self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

        self._frozen_names = ['_image_encoder', '_text_encoder']
        todd.reproduction.FrozenMixin.__init__(
            self,
            no_grad_config=dict(names=self._frozen_names),
            eval_config=dict(names=self._frozen_names),
        )

        self._register_state_dict_hook(self._state_dict_hook)

    def forward(self, batch: Batch, memo: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if memo is None or 'texts' not in memo:
            texts = self.encode_texts(batch)
            if memo is not None:
                memo['texts'] = texts
        else:
            texts = memo['texts']
        images = self.encode_images(batch)
        return (images @ texts.T) * self._scaler - self._bias

    def encode_images(self, batch: Batch) -> torch.Tensor:
        return self._image_encoder(batch)

    def encode_texts(self, batch: Batch) -> torch.Tensor:
        return self._text_encoder(batch, self._text_prompt)

    def dump_texts(self, batch: Batch) -> Dict[str, torch.Tensor]:
        return dict(
            embeddings=self.encode_texts(batch),
            scaler=self._scaler,
            bias=self._bias,
        )


class Runner(BaseRunner):

    def __init__(self, *args, **kwargs) -> None:
        if debug.CPU:
            clip_model, _ = clip.load(config.pretrained, 'cpu')
        else:
            clip_model, _ = clip.load(config.pretrained)
        clip_model.requires_grad_(False)
        super().__init__(*args, clip_model=clip_model, **kwargs)

    def _build_dataloader(
        self,
        *args,
        config: Optional[todd.Config],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        assert config is not None
        dataset = CocoClassification(clip_model=clip_model, **config.dataset)
        sampler = (
            None if debug.CPU or todd.get_world_size() == 1 else
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
        config: Optional[todd.Config],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> nn.Module:
        assert config is not None
        model = Model(clip_model=clip_model, **config).requires_grad_()
        if not debug.CPU:
            model = model.cuda()
        if todd.get_world_size() > 1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[torch.cuda.current_device()],
            )
        return model

    def _before_run(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_run(*args, **kwargs)
        memo.update(results=[])
        return memo

    def _run_iter(
        self, *args, i: int, batch: Batch, memo: Dict[str, Any], **kwargs,
    ) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        logits = self._model(batch, memo)
        memo['results'].append((logits.argmax(-1), batch.labels))

    def _after_run_iter(self, *args, i: int, batch: Batch, memo: Dict[str, Any], log: bool = False, **kwargs) -> Optional[bool]:
        if log and todd.get_rank() == 0:
            self._logger.info(
                f'Val Step [{i}/{len(self._dataloader)}]'
            )
        if log and debug.DRY_RUN:
            return True

    def _after_run(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        results: Iterator[Tuple[torch.Tensor, ...]] = zip(*memo['results'])
        if not debug.CPU and todd.get_world_size() > 1:
            results = tuple(  # all_gather must exec for all ranks
                map(all_gather, results),
            )
        if todd.get_rank() == 0:
            results = map(lambda result: torch.cat(result).cpu().numpy(), results)
            image_preds, image_labels = results
            acc = sklearn.metrics.accuracy_score(image_labels, image_preds)
            self._logger.info(f"acc {acc * 100:.3f}")
            return acc
        else:
            return -1

    @torch.no_grad()
    def dump(self) -> None:
        model = self._model
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        assert isinstance(model, Model)
        batch: Batch = next(iter(self._dataloader))
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        state_dict = model.dump_texts(batch)
        state_dict.update(
            names=self._dataset.classnames,
        ),
        torch.save(state_dict, self._work_dir / f'epoch_{self._epoch}_classes.pth')


class Trainer(TrainerMixin, Runner):

    def _build_train_dataloader(self, *args, **kwargs) -> Tuple[
        CocoClassification,
        Optional[torch.utils.data.distributed.DistributedSampler],
        torch.utils.data.DataLoader,
    ]:
        return self._build_dataloader(*args, **kwargs)

    def _build_train_fixtures(
        self,
        *args,
        config: Optional[todd.Config],
        **kwargs,
    ) -> None:
        assert config is not None
        self._criterion = todd.losses.LOSSES.build(config.loss)

    def _before_train(self, *args, **kwargs) -> Dict[str, Any]:
        memo = super()._before_train(self, *args, **kwargs)
        memo.update(
            record=-1,
        )
        return memo

    def _train_iter(self, *args, epoch: int, i: int, batch: Batch, memo: Dict[str, Any], **kwargs) -> None:
        if not debug.CPU:
            batch = Batch(*[x.cuda() for x in batch])
        logits = self._model(batch)
        loss: torch.Tensor = self._criterion(
            logits,
            batch.labels,
        )
        self._model.zero_grad()
        loss.backward()
        self._optimizer.step()
        memo['loss'] = loss.item()

    def _after_train_iter(
        self,
        *args,
        epoch: int,
        i: int,
        batch: Optional[Batch],
        memo: Dict[str, Any],
        log: bool = False,
        **kwargs,
    ) -> Optional[bool]:
        if log and todd.get_rank() == 0:
            self._logger.info(
                f'Epoch [{epoch}/{self._config.train.epoch}] '
                f'Train Step [{i}/{len(self._train_dataloader)}]'
                f'Loss {memo.pop("loss"):.3f}'
            )
        if log and debug.DRY_RUN:
            return True

    def _after_train_epoch(self, *args, epoch: int, memo: Dict[str, Any], **kwargs) -> None:
        if todd.get_rank() == 0:
            self.save_checkpoint(epoch=epoch)
        mAP = self.run()
        memo.update(
            record=max(memo['record'], mAP),
        )
        if todd.get_rank() == 0:
            self._logger.info(
                f"record {memo['record'] * 100:.3f}, "
            )

    def _after_train(self, *args, memo: Dict[str, Any], **kwargs) -> float:
        return memo['record']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('mode', choices=['train', 'val', 'dump'])
    parser.add_argument('name', type=str)
    parser.add_argument('config', type=pathlib.Path)
    parser.add_argument('--odps', action=todd.DictAction)
    parser.add_argument('--k8s', action=todd.DictAction)
    parser.add_argument('--override', action=todd.DictAction)
    parser.add_argument('--load', type=int)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--local_rank', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = todd.Config.load(args.config)
    if args.odps is not None:
        odps_init(args.odps)
    if args.k8s is not None:
        k8s_init(args.k8s)
    debug.init(config=config)
    if args.override is not None:
        for k, v in args.override.items():
            todd.setattr_recur(config, k, v)

    if args.local_rank is not None:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if not debug.CPU:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(todd.get_local_rank())

    todd.reproduction.init_seed(args.seed)

    init_dict = dict(name=args.name, config=config, load=args.load)
    if args.mode == 'train':
        runner = Trainer(**init_dict)
    else:
        runner = Runner(**init_dict)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'val':
        runner.run()
    else:
        runner.dump()
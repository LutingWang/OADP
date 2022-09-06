from typing import Any, Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import clip.model
import einops

import todd


class Prompt(todd.base.Module):
    _pad_embedding: torch.Tensor

    def __init__(
        self,
        *args,
        prompt: str,
        embedding: nn.Embedding,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompt_tokens = clip.tokenize([prompt])[0, 1:-1]
        with torch.no_grad():
            prompt_embedding: torch.Tensor = embedding(prompt_tokens)
            pad_embedding: torch.Tensor = embedding(prompt_tokens.new_zeros([]))
        self._prompt_embedding = nn.Parameter(prompt_embedding)
        self.register_buffer('_pad_embedding', pad_embedding, persistent=False)

    def __len__(self) -> int:
        return self._prompt_embedding.shape[0]

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_lengths: torch.Tensor,
        pad_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embedding = einops.repeat(
            self._prompt_embedding,
            'l d -> n l d',
            n=text_embeddings.shape[0],
        )
        pad_embedding = einops.repeat(
            self._pad_embedding,
            'd -> n l d',
            n=text_embeddings.shape[0],
            l=pad_length,
        )
        text_embeddings = torch.cat(
            [text_embeddings[:, :1], prompt_embedding, text_embeddings[:, 1:], pad_embedding],
            dim=1,
        )
        text_lengths = text_lengths + len(self)
        return text_embeddings, text_lengths


class Classnames(todd.base.Module):
    classname_embeddings: torch.Tensor

    def __init__(
        self,
        *args,
        classnames: Sequence[str],
        embedding: nn.Embedding,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        classnames = [classname.replace('_', ' ') + '.' for classname in classnames]
        classname_tokens = clip.tokenize(classnames)
        self.classname_lengths = classname_tokens.argmax(dim=-1)
        with torch.no_grad():
            classname_embeddings: torch.Tensor = embedding(classname_tokens)
        self.register_buffer('classname_embeddings', classname_embeddings, persistent=False)


class CLIPTextEncoder(todd.base.Module):

    def __init__(self, clip_model: clip.model.CLIP) -> None:
        super().__init__()
        self._transformer = clip_model.transformer
        self._pe = clip_model.positional_embedding
        self._ln = clip_model.ln_final
        self._proj = clip_model.text_projection

    def forward(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self._proj
        x = x / x.norm(dim=-1, keepdim=True)
        return x


class TextEncoder(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        prompt_kwargs: Sequence[todd.base.Config],
        classnames_kwargs: todd.base.Config,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._clip_text_encoder = CLIPTextEncoder(clip_model)
        self._prompts = todd.base.ModuleList(
            Prompt(
                embedding=clip_model.token_embedding,
                **kwargs,
            ) for kwargs in prompt_kwargs
        )
        self._classnames = Classnames(
            embedding=clip_model.token_embedding,
            **classnames_kwargs,
        )

        self._max_prompt_length = len(max(self._prompts, key=len))

    def __len__(self) -> int:
        return len(self._prompts)

    def forward(self) -> List[torch.Tensor]:
        embeddings = []
        for prompt in self._prompts:
            x, l = prompt(
                self._classnames.classname_embeddings,
                self._classnames.classname_lengths,
                pad_length=self._max_prompt_length - len(prompt),
            )
            x = self._clip_text_encoder.forward(x, l)
            embeddings.append(x)
        return embeddings


class FPN(todd.base.Module):

    def __init__(
        self,
        *args,
        in_channels_list: List[int],
        out_channels: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        self._fpn_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])

    def forward(self, inputs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor]:
        laterals: List[torch.Tensor] = [
            conv(input_) for conv, input_ in zip(self._lateral_convs, inputs)
        ]
        for top, bottom in zip(laterals[-1:0:-1], laterals[-2::-1]):
            bottom += F.interpolate(top, size=bottom.shape[2:], mode='nearest')
        outs = tuple(
            conv(lateral) for conv, lateral in zip(self._fpn_convs, laterals)
        )
        return outs


class ImageEncoder(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._clip_image_encoder = clip_model.visual
        # self._fpn = FPN(
        #     in_channels_list=[64, 256, 512, 1024, 2048],
        #     out_channels=256,
        # )
        # self._fpn_out = nn.Linear(256, clip_model.visual.attnpool.c_proj.out_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image, feats = self._clip_image_encoder(image)
        # feats: Tuple[torch.Tensor, ...] = self._fpn(feats)
        # feat = sum(f.sum(dim=(2, 3)) for f in feats)
        # feat = self._fpn_out(feat)
        # return feat
        image = image / image.norm(dim=-1, keepdim=True)
        return image


class CustomCLIP(todd.reproduction.FrozenMixin, todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        classnames: Sequence[str],
        frozen_config: todd.base.Config,
        **kwargs,
    ) -> None:
        todd.base.Module.__init__(self, *args, **kwargs)
        # self.image_encoder = ImageEncoder(clip_model.visual)
        self._image_encoder = ImageEncoder(
            clip_model=clip_model,
        )
        self._text_encoder = TextEncoder(
            clip_model=clip_model,
            prompt_kwargs=[
                dict(prompt='a photo of a'),
            ],
            classnames_kwargs=dict(classnames=classnames),
        )

        self._scaler = nn.Parameter(torch.full(self.shape, 20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.full(self.shape, 4.0), requires_grad=True)

        todd.reproduction.FrozenMixin.__init__(self, **frozen_config)

    @property
    def shape(self) -> torch.Size:
        return torch.Size([1, len(self._text_encoder)])

    def forward(self, image: torch.Tensor) -> List[torch.Tensor]:
        image_feature = self._image_encoder(image)
        text_features = self._text_encoder()
        outputs = []
        for i, text_feature in enumerate(text_features):
            output = image_feature @ text_feature.T
            outputs.append(output * self._scaler[0, i] - self._bias[0, i])
        return outputs

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state_dict: Dict[str, Any] = super().state_dict(*args, **kwargs)
        return {
            k: v for k, v in state_dict.items()
            if (
                not k.startswith('_image_encoder._clip_image_encoder')
                and not k.startswith('_text_encoder._clip_text_encoder')
                and not k.startswith('_text_encoder._classnames')
            )
        }

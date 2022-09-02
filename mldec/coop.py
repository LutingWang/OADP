from typing import Any, Dict, List, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import clip
import clip.model
import einops

import todd


class Prompt(todd.base.Module):

    def __init__(
        self,
        *args,
        prompt: str,
        embedding: nn.Embedding,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        prompt_tokens = clip.tokenize([prompt])
        with torch.no_grad():
            prompt_embedding: torch.Tensor = embedding(prompt_tokens)
        prompt_embedding = prompt_embedding[0, 1:-1, :]
        self._prompt_embedding = nn.Parameter(prompt_embedding)

    def __len__(self) -> int:
        return self._prompt_embedding.shape[0]

    def forward(
        self,
        text_embeddings: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_embedding = einops.repeat(
            self._prompt_embedding,
            'l d -> n l d',
            n=text_embeddings.shape[0],
        )
        text_embeddings = torch.cat(
            [text_embeddings[:, :1, :], prompt_embedding, text_embeddings[:, 1:, :]],
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


class TextEncoder(todd.base.Module):

    def __init__(
        self,
        *args,
        clip_model: clip.model.CLIP,
        prompt_kwargs: Dict[str, Any],
        classnames_kwargs: Dict[str, Any],
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._prompt = Prompt(
            embedding=clip_model.token_embedding,
            **prompt_kwargs,
        )
        self._classnames = Classnames(
            embedding=clip_model.token_embedding,
            **classnames_kwargs,
        )

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self) -> torch.Tensor:
        x: torch.Tensor = self._classnames.classname_embeddings
        l = self._classnames.classname_lengths
        x, l = self._prompt(x, l)
        x = x + self.positional_embedding[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self.transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self.text_projection
        return x


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
        self._model = clip_model.visual
        self._fpn = FPN(
            in_channels_list=[64, 256, 512, 1024, 2048],
            out_channels=256,
        )
        self._fpn_out = nn.Linear(256, clip_model.visual.attnpool.c_proj.out_features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x, feats = self._model(image)
        feats: Tuple[torch.Tensor, ...] = self._fpn(feats)
        feat = sum(f.sum(dim=(2, 3)) for f in feats)
        feat = self._fpn_out(feat)
        return feat


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
            prompt_kwargs=dict(prompt='a X photo of a X'),
            classnames_kwargs=dict(classnames=classnames),
        )
        self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

        todd.reproduction.FrozenMixin.__init__(self, **frozen_config)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image_features = self._image_encoder(image)
        text_features = self._text_encoder()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        output = image_features @ text_features.T
        return output * self._scaler - self._bias


# class ImageEncoder(nn.Module):  # TODO: train, requires_grad, init
#     def __init__(self, model: nn.Module) -> None:
#         super().__init__()
#         self._model = model
#         self._ladder = torchvision.models.resnet18(num_classes=1024)
#         self._adapts = todd.base.Workflow.build(
#             'adapts',
#             adapted=dict(
#                 type='Conv2d',
#                 fields=['feats'],
#                 kernel_size=1,
#                 parallel=[
#                     dict(in_channels=64, out_channels=64),
#                     dict(in_channels=256, out_channels=64),
#                     dict(in_channels=512, out_channels=128),
#                     dict(in_channels=1024, out_channels=256),
#                     dict(in_channels=2048, out_channels=512),
#                 ],
#             ),
#         )
#         self.register_module('adapts', todd.distillers.BaseDistiller.workflow_to_module(self._adapts))

#     def train(self, mode: bool = True):
#         super().train(mode)
#         self._model.eval()
#         return self

#     def requires_grad_(self, requires_grad: bool = True):
#         super().requires_grad_(requires_grad)
#         self._model.requires_grad_(False)
#         return self

#     def forward(self, image: torch.Tensor) -> torch.Tensor:
#         _, feats = self._model(image)
#         feats = self._adapts(dict(feats=feats))['adapted']
#         x = self._ladder.conv1(image)
#         x = self._ladder.bn1(x)
#         x = self._ladder.relu(x)
#         x = self._ladder.maxpool(x)

#         x = self._ladder.layer1(x + feats[0])
#         x = self._ladder.layer2(x + feats[1])
#         x = self._ladder.layer3(x + feats[2])
#         x = self._ladder.layer4(x + feats[3])

#         x = self._ladder.avgpool(x + feats[4])
#         x = torch.flatten(x, 1)
#         x = self._ladder.fc(x)

#         return x

#         # image = image.type(self._ladder.conv1.weight.dtype)
#         # image = self._ladder.relu1(self._ladder.bn1(self._ladder.conv1(image)))
#         # image = self._ladder.relu2(self._ladder.bn2(self._ladder.conv2(image)))
#         # image = self._ladder.relu3(self._ladder.bn3(self._ladder.conv3(image)))
#         # image = self._ladder.avgpool(image)
#         # image = self._ladder.layer1(feats[0] + image)
#         # image = self._ladder.layer2(feats[1] + image)
#         # image = self._ladder.layer3(feats[2] + image)
#         # image = self._ladder.layer4(feats[3] + image)
#         # image = self._ladder.attnpool(feats[4] + image)

#         # return x + image * self._gate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip.model

import todd

_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16  # cfg.TRAINER.COOP.N_CTX
        ctx_init = "a photo of a"  # cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            # if cfg.TRAINER.COOP.CSC:
            #     print("Initializing class-specific contexts")
            #     ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            # else:
            print("Initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"  # cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class ImageEncoder(nn.Module):  # TODO: train, requires_grad, init
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model
        self._ladder = torchvision.models.resnet18(num_classes=1024)
        self._adapts = todd.base.Workflow.build(
            'adapts',
            adapted=dict(
                type='Conv2d',
                fields=['feats'],
                kernel_size=1,
                parallel=[
                    dict(in_channels=64, out_channels=64),
                    dict(in_channels=256, out_channels=64),
                    dict(in_channels=512, out_channels=128),
                    dict(in_channels=1024, out_channels=256),
                    dict(in_channels=2048, out_channels=512),
                ],
            ),
        )
        self.register_module('adapts', todd.distillers.BaseDistiller.workflow_to_module(self._adapts))

    def train(self, mode: bool = True):
        super().train(mode)
        self._model.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(requires_grad)
        self._model.requires_grad_(False)
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, feats = self._model(image)
        feats = self._adapts(dict(feats=feats))['adapted']
        x = self._ladder.conv1(image)
        x = self._ladder.bn1(x)
        x = self._ladder.relu(x)
        x = self._ladder.maxpool(x)

        x = self._ladder.layer1(x + feats[0])
        x = self._ladder.layer2(x + feats[1])
        x = self._ladder.layer3(x + feats[2])
        x = self._ladder.layer4(x + feats[3])

        x = self._ladder.avgpool(x + feats[4])
        x = torch.flatten(x, 1)
        x = self._ladder.fc(x)

        return x

        # image = image.type(self._ladder.conv1.weight.dtype)
        # image = self._ladder.relu1(self._ladder.bn1(self._ladder.conv1(image)))
        # image = self._ladder.relu2(self._ladder.bn2(self._ladder.conv2(image)))
        # image = self._ladder.relu3(self._ladder.bn3(self._ladder.conv3(image)))
        # image = self._ladder.avgpool(image)
        # image = self._ladder.layer1(feats[0] + image)
        # image = self._ladder.layer2(feats[1] + image)
        # image = self._ladder.layer3(feats[2] + image)
        # image = self._ladder.layer4(feats[3] + image)
        # image = self._ladder.attnpool(feats[4] + image)

        # return x + image * self._gate


class CustomCLIP(nn.Module):
    def __init__(self, clip_model, classnames):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        # self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self._scaler = nn.Parameter(torch.tensor(20.0), requires_grad=True)
        self._bias = nn.Parameter(torch.tensor(4.0), requires_grad=True)

    def train(self, mode: bool = True):
        super().train(False)
        if mode:
            self.prompt_learner.train()
            self.image_encoder.train()
        return self

    def requires_grad_(self, requires_grad: bool = True):
        super().requires_grad_(False)
        if requires_grad:
            self.prompt_learner.requires_grad_()
            self.image_encoder.requires_grad_()
            self._scaler.requires_grad_()
            self._bias.requires_grad_()
        return self

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # logit_scale = self.logit_scale.exp()
        # logits = logit_scale * image_features @ text_features.t()

        # return logits

        output = image_features @ text_features.T
        return output * self._scaler - self._bias

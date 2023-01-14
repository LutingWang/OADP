from typing import Sequence

import clip
import clip.model
import einops
import todd
import torch
import torch.nn.functional as F
import tqdm

from ..base import coco, device, lvis

prompts = [
    "This is a {}",
    "There is a {}",
    "a photo of a {} in the scene",
    "a photo of a small {} in the scene",
    "a photo of a medium {} in the scene",
    "a photo of a large {} in the scene",
    "a photo of a {}",
    "a photo of a small {}",
    "a photo of a medium {}",
    "a photo of a large {}",
    "This is a photo of a {}",
    "This is a photo of a small {}",
    "This is a photo of a medium {}",
    "This is a photo of a large {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "There is one {} in the scene",
    "This is a {} in the scene",
    "This is the {} in the scene",
    "This is one {} in the scene",
    "This is one small {} in the scene",
    "This is one medium {} in the scene",
    "This is one large {} in the scene",
    "There is a small {} in the scene",
    "There is a medium {} in the scene",
    "There is a large {} in the scene",
    "There is a {} in the photo",
    "There is the {} in the photo",
    "There is one {} in the photo",
    "There is a small {} in the photo",
    "There is the small {} in the photo",
    "There is one small {} in the photo",
    "There is a medium {} in the photo",
    "There is the medium {} in the photo",
    "There is one medium {} in the photo",
    "There is a large {} in the photo",
    "There is the large {} in the photo",
    "There is one large {} in the photo",
    "There is a {} in the picture",
    "There is the {} in the picture",
    "There is one {} in the picture",
    "There is a small {} in the picture",
    "There is the small {} in the picture",
    "There is one small {} in the picture",
    "There is a medium {} in the picture",
    "There is the medium {} in the picture",
    "There is one medium {} in the picture",
    "There is a large {} in the picture",
    "There is the large {} in the picture",
    "There is one large {} in the picture",
    "This is a {} in the photo",
    "This is the {} in the photo",
    "This is one {} in the photo",
    "This is a small {} in the photo",
    "This is the small {} in the photo",
    "This is one small {} in the photo",
    "This is a medium {} in the photo",
    "This is the medium {} in the photo",
    "This is one medium {} in the photo",
    "This is a large {} in the photo",
    "This is the large {} in the photo",
    "This is one large {} in the photo",
    "This is a {} in the picture",
    "This is the {} in the picture",
    "This is one {} in the picture",
    "This is a small {} in the picture",
    "This is the small {} in the picture",
    "This is one small {} in the picture",
    "This is a medium {} in the picture",
    "This is the medium {} in the picture",
    "This is one medium {} in the picture",
    "This is a large {} in the picture",
    "This is the large {} in the picture",
    "This is one large {} in the picture",
]


class TextEncoder(todd.Module):

    def __init__(
        self,
        *args,
        categories: Sequence[str],
        clip_model: clip.model.CLIP,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._categories = categories
        self._token_embedding = clip_model.token_embedding
        self._transformer = clip_model.transformer
        self._pe = clip_model.positional_embedding
        self._ln = clip_model.ln_final
        self._proj = clip_model.text_projection

    def forward(self, prompt: str) -> torch.Tensor:
        tokens = clip.adaptively_tokenize(
            map(prompt.format, self._categories),
            device,
        )
        x: torch.Tensor = self._token_embedding(tokens)
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)]
        x = x @ self._proj
        x = F.normalize(x)
        return x


def main() -> None:
    categories = tuple(set(coco.all_ + lvis.all_))
    clip_model, _ = clip.load(
        'ViT-B/32',
        device=device,
        download_root='pretrained/clip',
    )
    text_encoder = TextEncoder(
        categories=categories,
        clip_model=clip_model,
    ).eval()

    with torch.no_grad():
        embeddings = sum(map(text_encoder, tqdm.tqdm(prompts))) / len(prompts)

    state_dict = dict(embeddings=embeddings, names=categories)
    torch.save(state_dict, 'data/prompts/vild.pth')


if __name__ == '__main__':
    main()

import argparse
import torch
import torch.nn.functional as F

import clip
import clip.model
import einops
import tqdm

import todd


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


class TextEncoder(todd.base.Module):

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

    def forward(
        self,
        x: torch.Tensor,
        l: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self._pe[:x.shape[1]]
        x = einops.rearrange(x, 'n l d -> l n d')
        x = self._transformer(x)
        x = einops.rearrange(x, 'l n d -> n l d')
        x = self._ln(x)
        x = x[torch.arange(x.shape[0]), l]
        x = x @ self._proj
        return F.normalize(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('pretrained')
    parser.add_argument('split')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    clip_model, _ = clip.load(args.pretrained, 'cpu')
    clip_model.requires_grad_(False)
    text_encoder = TextEncoder(
        clip_model=clip_model,
    )

    # class_names = getattr(datasets, args.split)
    class_names = ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol',
        'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna',
        'apple', 'applesauce', 'apricot', 'apron', 'aquarium',
        'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor',
        'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer',]

    embeddings_list = []
    with torch.no_grad():
        for prompt in tqdm.tqdm(prompts):
            tokens = clip.tokenize([
                prompt.format(class_name) for class_name in class_names
            ])
            embeddings = clip_model.token_embedding(tokens)
            lengths = tokens.argmax(dim=-1)
            embeddings = text_encoder(embeddings, lengths)
            embeddings_list.append(embeddings)

    state_dict = dict(
        embeddings=sum(embeddings_list) / len(embeddings_list),
        names=class_names,
    )
    torch.save(state_dict, 'data/coco/prompt/vild_coco.pth.tmp')

import clip
import clip.model
import torch
import torch.nn.functional as F
import tqdm

from ..base import coco, lvis

prompts = [
    "This is a {}", "There is a {}", "a photo of a {} in the scene",
    "a photo of a small {} in the scene",
    "a photo of a medium {} in the scene",
    "a photo of a large {} in the scene", "a photo of a {}",
    "a photo of a small {}", "a photo of a medium {}", "a photo of a large {}",
    "This is a photo of a {}", "This is a photo of a small {}",
    "This is a photo of a medium {}", "This is a photo of a large {}",
    "There is a {} in the scene", "There is the {} in the scene",
    "There is one {} in the scene", "This is a {} in the scene",
    "This is the {} in the scene", "This is one {} in the scene",
    "This is one small {} in the scene", "This is one medium {} in the scene",
    "This is one large {} in the scene", "There is a small {} in the scene",
    "There is a medium {} in the scene", "There is a large {} in the scene",
    "There is a {} in the photo", "There is the {} in the photo",
    "There is one {} in the photo", "There is a small {} in the photo",
    "There is the small {} in the photo", "There is one small {} in the photo",
    "There is a medium {} in the photo", "There is the medium {} in the photo",
    "There is one medium {} in the photo", "There is a large {} in the photo",
    "There is the large {} in the photo", "There is one large {} in the photo",
    "There is a {} in the picture", "There is the {} in the picture",
    "There is one {} in the picture", "There is a small {} in the picture",
    "There is the small {} in the picture",
    "There is one small {} in the picture",
    "There is a medium {} in the picture",
    "There is the medium {} in the picture",
    "There is one medium {} in the picture",
    "There is a large {} in the picture",
    "There is the large {} in the picture",
    "There is one large {} in the picture", "This is a {} in the photo",
    "This is the {} in the photo", "This is one {} in the photo",
    "This is a small {} in the photo", "This is the small {} in the photo",
    "This is one small {} in the photo", "This is a medium {} in the photo",
    "This is the medium {} in the photo", "This is one medium {} in the photo",
    "This is a large {} in the photo", "This is the large {} in the photo",
    "This is one large {} in the photo", "This is a {} in the picture",
    "This is the {} in the picture", "This is one {} in the picture",
    "This is a small {} in the picture", "This is the small {} in the picture",
    "This is one small {} in the picture",
    "This is a medium {} in the picture",
    "This is the medium {} in the picture",
    "This is one medium {} in the picture",
    "This is a large {} in the picture", "This is the large {} in the picture",
    "This is one large {} in the picture"
]


def main() -> None:
    categories = sorted(set(coco.all_ + lvis.all_))
    model, _ = clip.load_default()

    embeddings = []
    with torch.no_grad():
        for prompt in tqdm.tqdm(prompts):
            texts = map(prompt.format, categories)
            tokens = clip.adaptively_tokenize(texts)
            embedding = model.encode_text(tokens)
            embeddings.append(F.normalize(embedding))

    state_dict = dict(
        embeddings=sum(embeddings) / len(embeddings),
        names=categories,
    )
    torch.save(state_dict, 'data/prompts/vild.pth')


if __name__ == '__main__':
    main()

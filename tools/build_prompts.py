import einops
import torch
import torch.nn.functional as F
import tqdm
# from mmdet.datasets import LVISV1Dataset
import todd.tasks.natural_language_processing as nlp
from todd.models.modules import CLIPText

from oadp.categories import coco, lvis, objects365, v3det


def vild() -> None:
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

    tokenizer = nlp.tokenizers.CLIPTokenizer(
        bpe_path='pretrained/clip/clip_bpe.txt.gz',
    )

    model = CLIPText(out_features=512)
    model.load_pretrained('pretrained/clip/ViT-B-32.pt')
    model.requires_grad_(False)
    model.eval()
    model.cuda()

    names = sorted(set(coco.all_ + lvis.all_ + objects365.all_ + v3det.all_))

    embeddings = []
    with torch.no_grad():
        for prompt in tqdm.tqdm(prompts):
            texts = map(prompt.format, names)
            tokens = tokenizer.encodes(texts)
            tokens = tokens.cuda()
            x = model(tokens)
            eos = CLIPText.eos(tokens, x)
            embeddings.append(eos)
    embeddings_ = torch.stack(embeddings)
    embeddings_ = F.normalize(embeddings_, dim=-1)
    embeddings_ = einops.reduce(embeddings_, 'n ... -> ...', 'mean')

    state_dict = dict(embeddings=embeddings_, names=names)
    torch.save(state_dict, 'data/prompts/vild.pth')


# def detpro() -> None:
#     embeddings = torch.load('pretrained/detpro/iou_neg5_ens.pth', 'cpu')

#     # lvis annotations have a typo, which is fixed in mmdet
#     # we need to change it back, so that the names match
#     names: list[str] = list(LVISV1Dataset.METAINFO['classes'])
#     i = names.index('speaker_(stereo_equipment)')
#     names[i] = 'speaker_(stero_equipment)'

#     state_dict = dict(embeddings=embeddings, names=names)
#     torch.save(state_dict, 'data/prompts/detpro_lvis.pth')


def main() -> None:
    vild()
    # detpro()


if __name__ == '__main__':
    main()

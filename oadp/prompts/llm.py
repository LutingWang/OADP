import clip
import clip.model
import torch
import tqdm
import json
import torch.nn.functional as F

def gen_prompts(json_path: list[str], output_path) -> None:
    text_dict = {}
    for path in json_path:
        text_dict.update(json.load(open(path)))
    model, _ = clip.load_default()
    embeddings = []
    print('Generating prompts...')
    print(text_dict.keys())
    with torch.no_grad():
        for descriptions in tqdm.tqdm(text_dict.values()):
            tokens = clip.adaptively_tokenize(descriptions)
            embedding = model.encode_text(tokens)
            embeddings.append(torch.unsqueeze(F.normalize(embedding), 0))
    embeddings = torch.cat(embeddings, dim=0)
    embeddings = embeddings.permute(0, 2, 1)
    state_dict = dict(
        embeddings=embeddings,
        names=list(text_dict.keys()),
    )
    torch.save(state_dict, output_path)

if __name__ == '__main__':
    gen_prompts(['data/prompts/coco.json'], 'data/prompts/llm.pth')
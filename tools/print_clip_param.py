import torch
from clip.model import CLIP

def print_model_param(model_path):
    state_dict = torch.load(model_path, map_location="cpu").state_dict()
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    print("embed_dim", embed_dim)
    print("image_resolution", image_resolution)
    print("vision_layers", vision_layers)
    print("vision_width", vision_width)
    print("vision_patch_size", vision_patch_size)
    print("context_length", context_length)
    print("vocab_size", vocab_size)
    print("transformer_width", transformer_width)
    print("transformer_heads", transformer_heads)
    print("transformer_layers", transformer_layers)


if __name__ == "__main__":
    print_model_param('pretrained/clip/ViT-B-32.pt')
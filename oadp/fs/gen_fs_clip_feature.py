import torch
import clip
from tqdm import tqdm
from .gen_label_map import CropDataset

def extract_clip_features(clip_model_path, dataset):
    model = torch.load(clip_model_path, map_location="cuda")
    for data in tqdm(dataset):
        image_features = model.encode_image(data)
        



if __name__ == "__main__":
    extract_clip_features("clip_model.pth", CropDataset())
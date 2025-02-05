import torch


def extract_fs_model_weights(grounding_dino_path, clip_model_path, output_path):
    grounding_dino = torch.load(grounding_dino_path, map_location="cpu")
    clip_model = torch.load(clip_model_path, map_location="cpu")

    grounding_dino_state_dict = grounding_dino['state_dict']
    clip_model_state_dict = clip_model.state_dict()

    fs_model = {}
    for key, value in grounding_dino_state_dict.items():
        if key.startswith('language_model'):
            fs_model[key] = value

    for key, value in clip_model_state_dict.items():
        fs_model[f'clip_model.{key}'] = value

    torch.save(fs_model, output_path)

def merge_fs_weights(grounding_dino_path, fs_model_path, output_path):
    grounding_dino = torch.load(grounding_dino_path, map_location="cpu")
    fs_model = torch.load(fs_model_path, map_location="cpu")

    grounding_dino_state_dict = grounding_dino['state_dict']
    fs_model_state_dict = fs_model['state_dict']

    grounding_dino_fs = {}
    for key, value in grounding_dino_state_dict.items():
        if not key.startswith('language_model'):
            grounding_dino_fs[key] = value
    
    for key, value in fs_model_state_dict.items():
        if key.startswith('bert_model'):
            grounding_dino_fs[key] = value

    torch.save(grounding_dino_state_dict, output_path)
    
if __name__ == "__main__":
    grounding_dino_path = 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'
    clip_model_path = 'work_dirs/fs/epoch_7.pth'
    # merge_fs_weights(grounding_dino_path, clip_model_path, output_path)

    clip_model_path = 'pretrained/clip/ViT-B-32.pt'
    output_path = 'pretrained/oadp/fs/language_model.pth'
    extract_fs_model_weights(grounding_dino_path, clip_model_path, output_path)








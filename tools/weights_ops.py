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

def insert_ov_moe(grounding_dino_path, output_path, expert_num, insert_model='encoder'):
    grounding_dino = torch.load(grounding_dino_path, map_location="cpu")
    grounding_dino_state_dict = grounding_dino['state_dict']

    new_state_dict = {}
    for key, value in grounding_dino_state_dict.items():
        if key.startswith(insert_model) and 'ffn' in key:
            for i in range(expert_num):
                parts = key.split('.')
                ffnn_index = parts.index('ffn')
                new_parts = parts[:ffnn_index + 1] + ['experts', f'{i}'] + parts[ffnn_index + 1:]
                new_state_dict['.'.join(new_parts)] = value
        else:
            new_state_dict[key] = value
    grounding_dino_state_dict['state_dict'] = new_state_dict
    torch.save(grounding_dino_state_dict, output_path)

def weights_diff(weights_path_a, weights_path_b):
    a_model = torch.load(weights_path_a, map_location="cpu")
    b_model = torch.load(weights_path_b, map_location="cpu")
    a_keys = a_model['state_dict'].keys()
    b_keys = b_model['state_dict'].keys()
    for key in a_keys:
        if key not in b_keys:
            print(f'{key} not in b')
    for key in b_keys:
        if key not in a_keys:
            print(f'{key} not in a')


if __name__ == "__main__":
    grounding_dino_path = 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth'
    output_path = 'pretrained/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_moe.pth'
    gt_path = 'work_dirs/dp_o365_goldg_v3det_test/iter_1.pth'
    insert_ov_moe(grounding_dino_path, output_path, expert_num=4)
    weights_diff(gt_path, output_path)







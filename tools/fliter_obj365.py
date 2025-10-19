import json
import os
import tqdm

def fliter_obj365(ann_file):
    with open(ann_file, 'r') as f:
        data = []
        for line in f:
            item = json.loads(line)
            data.append(item)
    data_exist = []
    for instance in tqdm.tqdm(data):
        pth_name = os.path.basename(instance['filename']) + ".pth"
        objects_pth_path = os.path.join("data/oadp/objects_o365_eva-clip/train", pth_name)
        blocks_pth_path = os.path.join("data/oadp/blocks_o365_eva-clip/train", pth_name)
        globals_pth_path = os.path.join("data/oadp/globals_o365_eva-clip/train", pth_name)
        if os.path.exists(objects_pth_path) and os.path.exists(blocks_pth_path) and os.path.exists(globals_pth_path):
            data_exist.append(instance)
    with open("data/grounding_data/objects365/annotations/objects365_train.fs_od_exist.jsonl", "w") as f:
        for instance in data_exist:
            f.write(json.dumps(instance) + "\n")

if __name__ == "__main__":
    ann_file = "data/grounding_data/objects365/annotations/objects365_train.fs_od.json"
    fliter_obj365(ann_file)
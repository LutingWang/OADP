import os
import torch
from lvis import LVIS, LVISEval, LVISResults
from tqdm import tqdm
import pickle as pkl
import numpy as np
import json
import multiprocessing
from functools import partial

# These assignments are deprecated in newer numpy versions, but kept for compatibility.
# For future code, prefer using np.int32 and np.float32 directly.
np.long = np.int32
np.float = np.float32

def convert_pkl_to_coco(pkl_path):
    coco_data = {}
    with open(pkl_path, "rb") as f:
        results = pkl.load(f)
        for result in tqdm(results, desc=f"Loading {os.path.basename(pkl_path)}"):
            coco_data[result["img_id"]] = result["pred_instances"]
    return coco_data

def convert_preds_dir_to_coco(preds_dir_path):
    coco_data = {}
    file_list = [f for f in os.listdir(preds_dir_path) if f.endswith(".pth")]
    for file in tqdm(file_list, desc=f"Loading from {os.path.basename(preds_dir_path)}"):
        results = torch.load(os.path.join(preds_dir_path, file))
        img_id = int(os.path.basename(file).split("_")[-1].split(".")[0])
        coco_data[img_id] = results
    return coco_data

def merge_preds(coco_data_1, coco_data_2, w1, b1):
    def inv_sigmoid(x):
        # Clip values to avoid log(0) or division by zero
        x = np.clip(x, 1e-7, 1 - 1e-7)
        return np.log(x / (1 - x))

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    merged_data = {}
    
    # Add all predictions from first dictionary
    for img_id, pred in coco_data_1.items():
        merged_data[img_id] = pred.copy()
    
    # Merge predictions from second dictionary
    for img_id, pred2 in coco_data_2.items():
        if img_id in merged_data:
            pred1 = merged_data[img_id]
            merged_pred = {}
            
            if 'bboxes' in pred1 and 'bboxes' in pred2:
                merged_pred['bboxes'] = np.concatenate([pred1['bboxes'], pred2['bboxes']], axis=0)
                
            if 'scores' in pred1 and 'scores' in pred2:
                # Apply weighting only to the first set of scores
                scores1 = sigmoid(inv_sigmoid(pred1['scores']) * w1 + b1)
                scores2 = pred2['scores']
                merged_pred['scores'] = np.concatenate([scores1, scores2], axis=0)

            if 'labels' in pred1 and 'labels' in pred2:
                merged_pred['labels'] = np.concatenate([pred1['labels'], pred2['labels']], axis=0)
            
            # Copy other keys, prioritizing the first prediction set if there are conflicts
            for key in pred1:
                if key not in ['bboxes', 'scores', 'labels']:
                    merged_pred[key] = pred1[key]
            for key in pred2:
                if key not in ['bboxes', 'scores', 'labels'] and key not in merged_pred:
                    merged_pred[key] = pred2[key]
            
            merged_data[img_id] = merged_pred
        else:
            merged_data[img_id] = pred2.copy()

    return merged_data

def convert_to_lvis_format(pred_dict, lvis_api):
    lvis_results = []
    cat_ids = lvis_api.get_cat_ids()
    # Create a mapping from label index to LVIS category ID
    label_to_cat_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    
    for img_id, pred in pred_dict.items():
        if 'bboxes' not in pred or len(pred['bboxes']) == 0:
            continue
            
        bboxes = pred['bboxes']
        scores = pred['scores']
        labels = pred['labels']
        
        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            
            lvis_results.append({
                'image_id': int(img_id),
                'category_id': int(label_to_cat_id[labels[i]]),
                'bbox': bbox,
                'score': float(scores[i])
            })
    return lvis_results

def evaluate_lvis_results(gt_path, pred_results):
    """Note: This function prints to stdout, which can get messy with multiprocessing."""
    lvis_gt = LVIS(gt_path)
    lvis_pred_format = convert_to_lvis_format(pred_results, lvis_gt)
    
    if not lvis_pred_format:
        return {}

    lvis_results_obj = LVISResults(lvis_gt, lvis_pred_format, max_dets=-1)
    lvis_eval = LVISEval(lvis_gt, lvis_results_obj, iou_type='bbox')
    lvis_eval.run()
    # Suppress individual print_results() to keep the main output clean
    # lvis_eval.print_results() 
    
    metrics = {k: v for k, v in lvis_eval.results.items() if k.startswith('AP')}
    return metrics

def process_evaluation(params, gt_path, dp_fs_results, dp_ov_results):
    """
    Worker function for a single (w, b) evaluation.
    This function will be executed by each process in the pool.
    """
    w, b = params
    
    # Merge predictions with the current w and b values
    merged_results = merge_preds(dp_fs_results, dp_ov_results, w, b)
    
    # Evaluate the merged results
    metrics = evaluate_lvis_results(gt_path, merged_results)
    
    print(f"w: {w:.2f}, b: {b:.2f} -> AP: {metrics.get('AP', 'N/A'):.4f}")
    
    return {
        "w": w,
        "b": b,
        "metrics": metrics
    }


if __name__ == "__main__":
    # This check is CRUCIAL for multiprocessing to work correctly.
    multiprocessing.set_start_method("fork", force=True) # Use 'fork' for better memory efficiency on Linux/macOS
    
    # --- Configuration ---
    eval_type = "val"
    base_path = "data/grounding_data/coco"
    
    if eval_type == "minival":
        gt_path = "data/grounding_data/coco/annotations/lvis_v1_minival_inserted_image_name.json"
        dp_fs_path = "data/image_minival_results"
        dp_ov_path = "data/textual_minival_shortest_results"
    elif eval_type == "val":
        gt_path = "data/grounding_data/coco/annotations/lvis_v1_val.json"
        dp_fs_path = "data/image_val_results"
        dp_ov_path = "data/textual_val_shortest_results" 
    else:
        raise RuntimeError("wrong eval type")

    # --- Data Loading ---
    print("Loading prediction results...")
    dp_fs_results = convert_preds_dir_to_coco(dp_fs_path)
    dp_ov_results = convert_preds_dir_to_coco(dp_ov_path)

    # --- Parameter Grid ---
    w_list = [6]
    b_list = [-10]
    param_grid = [(w, b) for w in w_list for b in b_list]

    # --- Parallel Processing ---
    # Use all available CPU cores
    num_workers = 4
    print(f"\nStarting parameter search with {len(param_grid)} combinations using {num_workers} processes.")

    # Use functools.partial to fix arguments that are the same for every process
    worker_func = partial(process_evaluation, 
                          gt_path=gt_path, 
                          dp_fs_results=dp_fs_results, 
                          dp_ov_results=dp_ov_results)

    results = []
    # Create a pool of processes
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap to apply the worker function to the parameter grid
        # and wrap with tqdm for a progress bar
        pbar = tqdm(pool.imap(worker_func, param_grid), total=len(param_grid))
        for res in pbar:
            results.append(res)

    # --- Save Results ---
    print("\nParameter search complete. Saving results...")
    with open("results_multiprocess.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("Results saved to results_multiprocess.json")
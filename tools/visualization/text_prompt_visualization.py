from mmdet.datasets.lvis import LVISV1Dataset
import cv2
import supervision as sv
import numpy as np
from pycocotools.coco import COCO
import os
import pickle as pkl
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

lvis_class_names = LVISV1Dataset.METAINFO["classes"]


def convert_pkl_to_coco(pkl_path):
    coco_data = {}
    with open(pkl_path, "rb") as f:
        results = pkl.load(f)
        for result in tqdm(results):
            coco_data[result["img_id"]] = result["pred_instances"]
    return coco_data


def convert_ann_to_detections(anns, margin=0, threshold=0.55):
    if type(anns) == dict:
        mask = anns["scores"] > threshold
        boxes = anns["bboxes"][mask] # [x1, y1, x2, y2]
        boxes[:, 0] = boxes[:, 0] + margin
        boxes[:, 1] = boxes[:, 1] + margin
        boxes[:, 2] = boxes[:, 2] + margin
        boxes[:, 3] = boxes[:, 3] + margin
        class_ids = anns["labels"][mask]
        confidences = anns["scores"][mask]
        class_names = []
        for i in np.where(mask)[0]:
            class_names.append(anns["label_names"][i])
    else:
        boxes = []
        class_ids = []
        confidences = []
        class_names = []

        for ann in anns:
            # Get bounding box (COCO format: [x, y, width, height])
            bbox = ann["bbox"]
            x1, y1, w, h = bbox
            x1 = max(0, x1 + margin)
            y1 = max(0, y1 + margin)
            x2, y2 = x1 + w, y1 + h
            boxes.append([x1, y1, x2, y2])
            class_ids.append(ann["category_id"])
            confidences.append(1.0)
            class_names.append(lvis_class_names[ann["category_id"] - 1])

    return (
        sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            class_id=np.array(class_ids, dtype=int),
            confidence=np.array(confidences, dtype=np.float32),
        ),
        class_names,
    )

def get_ignore_mask(class_names, name_ignore_list=[]):
    mask = []
    for class_name in class_names:
        if class_name not in name_ignore_list:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def label_result(image, anns, box_annotator, label_annotator):
    margin = 400
    image = cv2.copyMakeBorder(
        image,
        top=margin,
        bottom=margin,
        left=margin,
        right=margin,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    cv2.imwrite(f"work_dirs/visualization/annotated_image_test.png", image)
    detections, class_names = convert_ann_to_detections(anns, margin=margin)
    ignore_mask = get_ignore_mask(class_names, name_ignore_list=["person", "wall clock", "timer", "dollhouse", "deadbolt", "tachometer", "compass", "alarm clock", "barrel", "spectacles", "goggles", "lamppost", "brake light", "headlight", "convertible ", "cov", "quesadilla", "goat", "snowman", "wagon wheel", "toy", "scarecrow", "sherbert", "arctic ", "cub ", "giant panda", "pelican", "puffin", "dinghy", "raft", "kayak", "barge"])
    detections = detections[ignore_mask]
    class_names = [class_name for class_name, mask in zip(class_names, ignore_mask) if mask]
    if 1 in detections.confidence:
        labels = [
        f"{class_name}"
        for class_name, confidence in zip(class_names, detections.confidence)
    ]
    else:
        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, detections.confidence)
        ]
    annotated_image = box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )
    return annotated_image


def visualize_lvis_results(gt_path, pred_results, image_base_path, image_ids=None, max_images=None, output_path=None):
    # Load all result files
    gt_coco = COCO(gt_path)
    print("Loading done")

    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(
        text_scale=1,
        text_padding=1,
        text_position=sv.Position.TOP_RIGHT,
        smart_position=True,
    )

    # Get all unique image IDs from all result files
    if image_ids is None:
        all_image_ids = list(set(gt_coco.imgToAnns.keys()) & set(pred_results[0].keys()))
    else:
        all_image_ids = image_ids

    filtered_results = []
    for image_id in tqdm(all_image_ids[:max_images]):
        coco_path = gt_coco.imgs[image_id]["coco_url"].split(
            "http://images.cocodataset.org/"
        )[-1]
        image_path = os.path.join(image_base_path, coco_path)
        image = cv2.imread(image_path)

        # Create matplotlib figure with subplots for each result file
        num_results = len(pred_results) + 1
        fig, axes = plt.subplots(1, num_results, figsize=(6 * num_results, 6), gridspec_kw={'wspace': 0.05})

        # Process each result file
        gt_anns = gt_coco.imgToAnns[image_id]
        results = [gt_anns] + [res[image_id] for res in pred_results]
        filtered_results.append([image_path, results])
        for idx, result in enumerate(results):
            anns = result
            annotated_image = label_result(image, anns, box_annotator, label_annotator)
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            axes[idx].imshow(annotated_image_rgb)
            # axes[idx].set_title(f'Result {idx+1}: {os.path.basename(result_path)}')
            axes[idx].axis("off")

        plt.tight_layout(pad=0.1)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(
                os.path.join(output_path, f"{image_id}.pdf"),
                dpi=150,
                bbox_inches="tight",
            )

        plt.close()

    with open(os.path.join(output_path, "filtered_results.pkl"), "wb") as f:
        pkl.dump(filtered_results, f)


def visualize_filtered_results(filtered_results, output_path):
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(
        text_scale=1,
        text_padding=1,
        text_position=sv.Position.TOP_RIGHT,
        smart_position=True,
    )

    for image_idx, (image_path, results) in enumerate(filtered_results):
        image = cv2.imread(image_path)
        # _, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 6), gridspec_kw={'wspace': 0.01})
        # breakpoint()
        for ann_idx, anns in enumerate(results):
            annotated_image = label_result(image, anns, box_annotator, label_annotator)
            # annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            # axes[ann_idx].imshow(annotated_image_rgb)
            # axes[ann_idx].axis("off")
            cv2.imwrite(os.path.join(output_path, f"{image_idx}_{ann_idx}.png"), annotated_image)
        
        # plt.tight_layout(pad=0.1)
        # if output_path is not None:
        #     os.makedirs(output_path, exist_ok=True)
        #     plt.savefig(
        #         os.path.join(output_path, f"{image_idx}.pdf"),
        #         dpi=150,
        #         bbox_inches="tight",
        #     )
    plt.close()


if __name__ == "__main__":
    # base_path = "data/grounding_data/coco"
    # gt_path = "data/grounding_data/coco/annotations/lvis_v1_minival_inserted_image_name.json"
    # gd_path = "work_dirs/ov_distill_0.025_0.25_0.025/new_iter_150000/lvis_minival_gd.pkl"
    # dp_gd_path = "work_dirs/ov_distill_0.025_0.25_0.025/new_iter_150000/lvis_minival.pkl"
    # print("Loading results...")
    # gd_results = convert_pkl_to_coco(gd_path)
    # dp_gd_results = convert_pkl_to_coco(dp_gd_path)

    # os.makedirs("data/visualization", exist_ok=True)
    # image_ids = [519491, 475904, 458755, 335954, 196759, 42889, 2149]
    # visualize_lvis_results(
    #     gt_path, [dp_gd_results, gd_results], base_path, image_ids=image_ids, max_images=-1, output_path="work_dirs/visualization/"
    # )
    filtered_results = pkl.load(open("work_dirs/visualization/filtered_results.pkl", "rb"))
    visualize_filtered_results(filtered_results, "work_dirs/visualization/")
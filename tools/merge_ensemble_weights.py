import torch


def merge_ensemble_weights(text_detector_weights_path, image_detector_weights_path):
    text_detector_weights = torch.load(text_detector_weights_path, "cpu")
    image_detector_weights = torch.load(image_detector_weights_path, "cpu")
    merged_weights = {}
    merged_weights["state_dict"] = {}
    for key in text_detector_weights["state_dict"].keys():
        merged_weights["state_dict"][f"text_detector.{key}"] = text_detector_weights[
            "state_dict"
        ][key]
    for key in image_detector_weights["state_dict"].keys():
        merged_weights["state_dict"][f"image_detector.{key}"] = image_detector_weights[
            "state_dict"
        ][key]
    return merged_weights


if __name__ == "__main__":
    text_detector_weights_path = (
        "work_dirs/ov_distill_0_0.25_0_global_block_object_no_lcs/iter_150000.pth"
    )
    image_detector_weights_path = (
        "work_dirs/fs_llm_features_distill_0.00_0.8_0.0_lvis_fine/iter_16000.pth"
    )
    merged_weights = merge_ensemble_weights(
        text_detector_weights_path, image_detector_weights_path
    )
    torch.save(merged_weights, "work_dirs/ensemble/ensemble_weights.pth")

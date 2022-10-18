from typing import Tuple
import todd
import torch

from mmdet.datasets import build_dataset
from mmdet.core import bbox2result

import cafe


def _soft_nms(
    box_class,
    pairwise_iou_func,
    boxes,
    scores,
    gaussian_sigma,
    prune_threshold,
):
    boxes = boxes.clone()
    scores = scores.clone()
    idxs = torch.arange(scores.size()[0])

    idxs_out = []
    scores_out = []

    while scores.numel() > 0:
        top_idx = torch.argmax(scores)
        idxs_out.append(idxs[top_idx].item())
        scores_out.append(scores[top_idx].item())

        top_box = boxes[top_idx]
        ious = pairwise_iou_func(box_class(top_box.unsqueeze(0)), box_class(boxes))[0]

        decay = torch.exp(-torch.pow(ious, 2) / gaussian_sigma)

        scores *= decay
        keep = scores > prune_threshold
        keep[top_idx] = False

        boxes = boxes[keep]
        scores = scores[keep]
        idxs = idxs[keep]

    return torch.tensor(idxs_out).to(boxes.device), torch.tensor(scores_out).to(scores.device)


def batched_soft_nms(
    boxes, scores, idxs, gaussian_sigma, prune_threshold
):
    assert boxes.numel() > 0
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    return _soft_nms(
        todd.BBoxesXYXY,
        todd.BBoxesXYXY.ious,
        boxes_for_nms, scores, gaussian_sigma, prune_threshold
    )


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    image_wh: Tuple[int, int],
):
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = todd.BBoxesXYXY(boxes.reshape(-1, 4)).clamp(image_wh)
    boxes = boxes.to_tensor().view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > 0.001  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep, soft_nms_scores = batched_soft_nms(
        boxes,
        scores,
        filter_inds[:, 1],
        0.5,
        0.001,
    )
    scores[keep] = soft_nms_scores
    keep = keep[:100]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    return boxes, scores, filter_inds[:, 1]


config = todd.Config.load('configs/cafe/faster_rcnn/cafe_48_17.py')
access_layer = todd.datasets.PthAccessLayer(
    data_root='../RegionCLIP/data/coco/regionclip_preds',
)
dataset = build_dataset(config.data.test, default_args=dict(test_mode=True))
results = []
for image_id in dataset.img_ids:
    data = access_layer[f'{image_id:012d}']
    scores = torch.cat(
        (data['scores'], torch.zeros_like(data['scores'][:, [0]])),
        dim=-1,
    )
    bboxes, scores, classes = fast_rcnn_inference_single_image(
        data['bboxes'].float(),
        scores.float().softmax(-1),
        data['input_wh'],
    )
    bboxes = todd.BBoxesXYXY(bboxes).scale(
        ratio_wh=(
            data['image_wh'][0] / data['input_wh'][0],
            data['image_wh'][1] / data['input_wh'][1],
        ),
    ).clamp(image_wh=data['image_wh'])
    indices = bboxes.indices(min_wh=(0, 0))
    bboxes = bboxes[indices].to_tensor()
    scores = scores[indices]
    classes = classes[indices]
    results.append(
        bbox2result(
            torch.cat((bboxes, scores.unsqueeze(-1)), dim=-1),
            classes,
            65,
        ),
    )
result = dataset.evaluate(results)
print(result)

'''
Instances(num_instances=100, image_height=426, image_width=640, fields=[pred_boxes: Boxes(tensor([[292.3128, 215.9288, 351.4143, 318.4350],
        [409.9833, 156.6881, 466.4226, 299.5313],
        [  2.4251, 167.6044, 154.1431, 266.9156],
        [363.9268, 219.6563, 415.8402, 317.9025],
        [354.0765, 213.2663, 362.0632, 231.3713],
        [404.9251, 212.4675, 442.7288, 308.8500],
        [166.7887, 232.9688, 186.2230, 266.7825],
        [556.9384, 211.4025, 640.0000, 289.6800],
        [442.9950, 168.0038, 514.3428, 283.5563],
        [242.9285, 198.2231, 254.2429, 212.7338],
        [360.4659, 214.5975, 373.7770, 230.9719],
        [383.6273, 171.5981, 401.7304, 211.6688],
        [550.5491, 301.3950, 587.8203, 399.9075],
        [490.6489, 175.0594, 514.0765, 285.1538],
        [306.9551, 215.6625, 342.6289, 233.5013],
        [ 46.2230, 213.1331,  61.3311, 239.6250],
        [314.9417, 219.6563, 349.8170, 255.6000],
        [447.5208, 120.1453, 461.0981, 142.3106],
        [318.6689, 203.6813, 325.8569, 215.7956],
        [243.0616, 202.4831, 250.5158, 213.7988],
        [351.1481, 205.0125, 358.8685, 218.7244],
        [357.8036, 236.5631, 384.4259, 318.9675],
        [460.5657, 159.2175, 469.6173, 168.5363],
        [432.6123, 229.1081, 446.7221, 306.1875],
        [354.6090, 217.1269, 361.2646, 231.1050],
        [383.0948, 219.3900, 413.7105, 305.1225],
        [513.0117, 222.8513, 542.8286, 278.4975],
        [550.0166, 299.2650, 587.2878, 398.3100],
        [234.9418, 197.1581, 242.2629, 211.9350],
        [315.7404, 248.6775, 351.6805, 316.0388],
        [166.3893, 232.1700, 186.2230, 267.7144],
        [382.8286, 155.8894, 461.0981, 297.4013],
        [339.1680, 204.2138, 347.4210, 213.6656],
        [349.5508, 202.0838, 363.6606, 231.6375],
        [492.7787, 157.3538, 499.1680, 173.9944],
        [413.4442, 215.7956, 439.5341, 266.2500],
        [314.6755, 216.9938, 338.3694, 230.0400],
        [577.7038, 250.5413, 604.8585, 265.5844],
        [512.7454, 224.0494, 543.0948, 278.4975],
        [339.1680, 244.2844, 354.8752, 313.6425],
        [564.9251, 211.0031, 637.8702, 277.9650],
        [606.9883, 287.0175, 639.4675, 307.5188],
        [436.8719, 232.3031, 445.3910, 288.0825],
        [293.3777, 218.7244, 329.5840, 309.6488],
        [317.0715, 193.2975, 323.1947, 214.0650],
        [483.4609, 360.5025, 638.4026, 411.8888],
        [327.4542, 216.0619, 352.7454, 229.3744],
        [347.4210, 204.8794, 354.6090, 216.3281],
        [454.9750, 169.0688, 497.3045, 288.6150],
        [384.9584, 171.7313, 400.1331, 210.3375],
        [240.9318, 192.0994, 252.7787, 213.3994],
        [461.3644, 367.9575, 549.4842, 426.0000],
        [358.3361, 214.9969, 374.3095, 230.9719],
        [312.5457, 204.2138, 318.9351, 216.7275],
        [384.4259, 262.5225, 413.4442, 313.1100],
        [497.0383, 158.5519, 502.8952, 174.2606],
        [360.4659, 214.9969, 373.7770, 231.3713],
        [141.0982, 284.3550, 190.4825, 291.5438],
        [432.6123, 229.6406, 463.4942, 305.9213],
        [  3.3652, 161.6138, 156.2729, 261.4575],
        [346.0898, 213.1331, 353.2779, 231.5044],
        [242.7953, 204.3469, 251.4476, 212.7338],
        [317.6040, 221.7863, 342.6289, 233.5013],
        [353.8103, 221.3869, 362.3294, 230.7056],
        [375.6406, 219.2569, 411.8469, 253.4700],
        [349.8170, 197.4244, 359.6672, 214.8638],
        [408.1198, 157.7531, 454.1764, 223.5169],
        [397.7371, 218.8575, 418.7687, 307.5188],
        [457.9035, 159.3506, 468.8186, 167.7375],
        [330.1165, 226.9781, 382.2962, 315.7725],
        [219.6339,   6.0489, 258.7687,  38.9058],
        [338.3694, 201.2850, 346.3560, 216.1950],
        [338.1031, 221.5200, 345.5574, 232.3031],
        [474.4093, 136.4531, 523.9268, 174.2606],
        [411.5807, 213.1331, 441.1314, 305.1225],
        [489.3178, 173.0625, 510.8818, 284.8875],
        [462.6955, 215.5294, 473.3444, 226.1794],
        [166.2562, 233.5013, 186.0898, 267.4482],
        [396.1398, 210.6038, 446.7221, 312.8438],
        [494.9085, 154.6913, 502.0965, 171.8644],
        [363.3943, 248.9438, 383.0948, 317.3700],
        [422.7621, 175.7250, 444.5923, 238.1606],
        [147.3544, 278.7638, 188.2196, 285.4200],
        [ 46.8552, 213.5325,  59.7671, 239.6250],
        [368.4526, 232.7025, 385.7570, 246.8138],
        [449.3843, 120.0122, 462.1631, 143.1094],
        [303.7604, 216.4613, 343.9601, 244.8169],
        [465.6239, 208.7400, 488.2529, 282.4913],
        [242.7953, 204.3469, 251.4476, 212.7338],
        [351.6805, 201.9506, 359.9334, 228.4425],
        [314.4093, 193.4306, 321.3311, 213.9319],
        [549.4842, 300.5963, 587.2878, 394.3163],
        [366.0565, 242.0213, 384.6922, 253.7363],
        [237.7371, 189.1706, 262.7621, 211.9350],
        [332.7787, 208.0744, 350.3494, 215.6625],
        [554.2762, 207.5419, 638.9351, 344.2613],
        [218.5690, 230.1731, 293.3777, 310.9800],
        [179.7005, 276.6338, 193.4110, 284.0888],
        [606.4559, 300.5963, 640.0000, 355.1775],
        [ 46.8552, 213.5325,  59.7671, 239.6250]])), scores: tensor([0.9979, 0.9956, 0.9954, 0.9856, 0.9502, 0.9343, 0.9118, 0.9112, 0.9088,
        0.8978, 0.8889, 0.8445, 0.7169, 0.6956, 0.6408, 0.6397, 0.5608, 0.5325,
        0.5304, 0.5222, 0.4943, 0.4493, 0.4482, 0.4420, 0.4015, 0.3918, 0.3858,
        0.3719, 0.3384, 0.3205, 0.3106, 0.3069, 0.3064, 0.3039, 0.2924, 0.2870,
        0.2858, 0.2855, 0.2728, 0.2672, 0.2670, 0.2625, 0.2611, 0.2574, 0.2487,
        0.2426, 0.2389, 0.2324, 0.2278, 0.2256, 0.2162, 0.2135, 0.2131, 0.2116,
        0.2103, 0.2100, 0.2075, 0.2000, 0.1994, 0.1954, 0.1886, 0.1808, 0.1795,
        0.1778, 0.1752, 0.1689, 0.1687, 0.1633, 0.1618, 0.1594, 0.1582, 0.1493,
        0.1487, 0.1444, 0.1415, 0.1388, 0.1383, 0.1366, 0.1342, 0.1341, 0.1324,
        0.1316, 0.1303, 0.1303, 0.1297, 0.1264, 0.1213, 0.1209, 0.1174, 0.1153,
        0.1151, 0.1150, 0.1134, 0.1122, 0.1121, 0.1116, 0.1116, 0.1113, 0.1096,
        0.1092]), pred_classes: tensor([46,  0, 50, 46, 62, 46, 62, 50, 59, 62, 62,  0, 62, 59, 46, 31, 46, 61,
        62, 62, 62, 46,  0, 46, 62, 46, 56, 31, 62, 46, 32,  0, 62, 62, 31, 46,
        46, 44, 59, 46, 50, 43, 46, 46, 62, 30, 46, 62, 59,  0, 62, 46, 62, 62,
        46, 31, 32, 60, 46, 50, 62, 62, 46, 62, 46, 62,  0, 46,  0, 46, 36, 62,
        46, 55, 46, 59, 53, 62,  0, 31, 46, 22, 60,  0, 46, 61, 46, 59, 32, 62,
        62, 62, 46, 62, 62, 50, 46, 60, 46, 31])])
'''

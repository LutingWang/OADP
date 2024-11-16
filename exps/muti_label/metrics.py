import cv2
import os
import torch
import numpy as np
import mmcv
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS
from mmengine.visualization import Visualizer
from mmengine.dist import collect_results

from exps.muti_label.globals import cur_cates


@METRICS.register_module()
class MutiLabelMetric(BaseMetric):

    default_prefix = 'MutiLabel' 

    def __init__(self, threshold, collect_device = 'cpu', prefix = None, collect_dir = None):
        super().__init__(collect_device, prefix, collect_dir)
        self.threshold = threshold # threshold for classification prediction
        self.is_visualize = True

    def _average_precision(sekf, output: np.ndarray, target: np.ndarray) -> float:
        epsilon = 1e-8

        # sort examples
        indices = output.argsort()[::-1]
        # Computes prec@i
        total_count_ = np.cumsum(np.ones((len(output), 1)))

        target_ = target[indices]
        ind = target_ == 1
        pos_count_ = np.cumsum(ind)
        total = pos_count_[-1]
        pos_count_[np.logical_not(ind)] = 0
        pp = pos_count_ / total_count_
        precision_at_i_ = np.sum(pp)
        precision_at_i = precision_at_i_ / (total + epsilon)

        return precision_at_i

    def get_mAP(self, gts, preds):
        APs = []
        _, num_classes = gts.shape
        APs = np.zeros(num_classes)
        for k in range(num_classes):  # AP for each class
            APs[k] = self._average_precision(preds[:, k], gts[:, k])
        return APs.mean()

    def select_cates(self, preds: torch.Tensor):
        cates = [cur_cates[i] for i in range(len(cur_cates)) if preds[i]]
        return ', '.join(cates)

    def visualize(self, data_samples: list[dict], pred_labels: torch.Tensor):
        self.visualizer: Visualizer = Visualizer.get_current_instance()
        for sample, pred in zip(data_samples[0]['data_samples'], pred_labels):
            img = mmcv.imread(sample.img_path, channel_order='rgb')
            img_name = os.path.basename(sample.img_path)
            self.visualizer.set_image(img)
            text = self.select_cates(pred.cpu().numpy())
            self.visualizer.draw_texts(text, torch.tensor([10, 20]))
            self.visualizer.add_image(img_name, self.visualizer.get_image())

    def process(self, data_batch: list[dict], data_samples: list[dict]):
        pred_label = data_samples[0]['pred_logits']
        gt_label = torch.cat([sample.gt_label.unsqueeze(0) for sample in data_samples[0]['data_samples']], dim=0)
        
        result = {
            'pred': pred_label.cpu().numpy(),
            'gt': gt_label.cpu().numpy(),
            'img_path': [sample.img_path for sample in data_samples[0]['data_samples']]
        }
        self.results.append(result)

        if self.is_visualize:
            self.visualize(data_samples, pred_label > self.threshold)
            self.is_visualize = False

    def compute_metrics(self, results: list[dict]) -> dict:
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])
        mAP = self.get_mAP(gts, preds)


        pred_results = {}
        for batch in results:
            batch_imgs = batch['img_path']
            batch_preds = batch['pred']
            for img, pred in zip(batch_imgs, batch_preds):
                pred_results[img] = pred
        torch.save(pred_results, 'pred_results.pth')
        
        
        return {'mAP': mAP}
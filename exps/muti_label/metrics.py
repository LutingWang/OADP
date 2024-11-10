import numpy as np
from sklearn.metrics import accuracy_score, recall_score

from mmengine.logging import MMLogger
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS

from exps.muti_label.globals import cur_cates

@METRICS.register_module()
class MutiLabelMetric(BaseMetric):

    default_prefix = 'MutiLabel' 

    def __init__(self, threshold, collect_device = 'cpu', prefix = None, collect_dir = None):
        super().__init__(collect_device, prefix, collect_dir)
        self.threshold = threshold # threshold for classification prediction

    def process(self, data_batch: list[dict], data_samples: list[dict]):
        """Process the data batch and store the classification prediction results"""
        pred_label = (data_samples[0]['pred_logits'] > self.threshold)
        gt_label = data_samples[0]['gt_label']

        # d_label = np.argwhere(pred_label[0].cpu().numpy() == 1)[0]
        # print([cur_cates[i] for i in d_label])

        # fetch classification prediction results and category labels
        result = {
            'pred': pred_label.cpu().numpy(),
            'gt': gt_label.cpu().numpy()
        }

        # store the results of the current batch into self.results
        self.results.append(result)

    def compute_metrics(self, results: list[dict]) -> dict:
        """Compute the metrics from processed results.

        Args:
            results (dict): The processed results of each batch.

        Returns:
            Dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """

        # aggregate the classification prediction results and category labels for all samples
        preds = np.concatenate([res['pred'] for res in results])
        gts = np.concatenate([res['gt'] for res in results])
        accuracy = accuracy_score(gts, preds)
        recall = recall_score(gts, preds, average='macro')
        # # log the classification report
        results =  {
            'accuracy': accuracy,
            'recall': recall
        }
        return results
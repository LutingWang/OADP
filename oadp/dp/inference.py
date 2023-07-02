import todd
import numpy as np
import torch

from typing import Any
from mmcv.parallel import collate
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def single_gpu_infer(model: Any, imgs: str, config: todd.Config):
    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    validator: todd.Config = config.validator.dataloader
    validator.dataset.pipeline = replace_ImageToTensor(validator.dataset.pipeline)
    test_pipeline = Compose(validator.dataset.pipeline)

    data_list = []
    for img in imgs:
        if isinstance(img, np.ndarray):
            data = dict(img=img)
        else:
            data = dict(img_info=dict(filename=img), img_prefix=None)
        data = test_pipeline(data)
        data_list.append(data)
    
    data = collate(data_list, samples_per_gpu=len(imgs))
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    if not is_batch:
        return results[0]
    else:
        return results
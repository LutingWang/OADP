# mypy: disable-error-code="misc"

__all__ = [
    'COCOGlobalDataset',
]
from PIL import Image

from todd.patches.pil.image import convert_rgb
from todd.datasets.coco import T
from todd.datasets import COCODataset

from ..registries import OAKEDatasetRegistry
from .datasets import GlobalDataset
from ..dataset.odvg import ODVGDataset

@OAKEDatasetRegistry.register_()
class COCOGlobalDataset(GlobalDataset, COCODataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)


@OAKEDatasetRegistry.register_()
class V3DetGlobalDataset(ODVGDataset):
    DATA_ROOT = 'data/V3Det'
    ANNOTATIONS_FILE = 'annotations/v3det_2023_v1_train_od.json'
    IMAGE_ROOT = ''
    LABEL_MAP = 'annotations/v3det_2023_v1_label_map.json'

    def __init__(self, *args, auto_fix, split, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def bind(self, runner) -> None:
        self.runner = runner
        self._transforms = self.runner.transforms
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        img_path = data['img_path']
        image = convert_rgb(Image.open(img_path))
        tensor = self._transforms(image)
        key = data['filename'].replace('.jpg', '').replace('/','-')
        return T(id_=key, image=tensor, annotations=data['instances'])


@OAKEDatasetRegistry.register_()
class Objects365v1GlobalDataset(V3DetGlobalDataset):
    DATA_ROOT = 'data/objects365v1'
    ANNOTATIONS_FILE = 'objects365_train_od.json'
    IMAGE_ROOT = 'train'
    LABEL_MAP = 'o365v1_label_map.json'
    

@OAKEDatasetRegistry.register_()
class FlikerGlobalDataset(V3DetGlobalDataset):
    DATA_ROOT = 'data/flickr30k_entities'
    ANNOTATIONS_FILE = 'final_flickr_separateGT_train_vg.json'
    IMAGE_ROOT = 'flickr30k_images'
    LABEL_MAP = None


@OAKEDatasetRegistry.register_()
class GQAGlobalDataset(V3DetGlobalDataset):
    DATA_ROOT = 'data/gqa'
    ANNOTATIONS_FILE = 'final_mixed_train_no_coco_vg.json'
    IMAGE_ROOT = 'images'
    LABEL_MAP = None
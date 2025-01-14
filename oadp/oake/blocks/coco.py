__all__ = [
    "COCOBlockDataset",
]

from todd.datasets import COCODataset
import torchvision.transforms.v2 as tf_v2
from ..registries import OAKEDatasetRegistry
from .datasets import *
from ..globals_.coco import V3DetGlobalDataset


@OAKEDatasetRegistry.register_()
class COCOBlockDataset(BlockDatasetMixin, COCODataset):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, load_annotations=False, **kwargs)

    def _getitem(self, index: int) -> Batch:
        key, image = self._access(index)
        bboxes, blocks = self._partition(image)
        return Batch(id_=key, bboxes=bboxes, blocks=blocks)


@OAKEDatasetRegistry.register_()
class BaseBlockDataset(V3DetGlobalDataset):

    def __init__(
        self,
        *args,
        block_size: int = 224,
        max_stride: int = 112,
        rescale: float = 1.5,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._partitioner = Partitioner(block_size, max_stride)
        self._rescale = rescale
        # self._transform = tf_v2.Compose([
        #     tf_v2.Resize(size=[224], interpolation=tf_v2.InterpolationMode.BICUBIC),
        #     tf_v2.CenterCrop(size=(224, 224)),
        #     tf_v2.ToImage(),
        #     tf_v2.ToDtype(torch.float32, True),
        #     tf_v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711], inplace=False),
        # ])

    def _partition(
        self,
        image: Image.Image,
    ) -> tuple[BBoxes, torch.Tensor]:
        w, h = image.size
        if w > h:
            offset = (w - h) / 2
            bbox = (offset, 0.0, h + offset, h)
        else:
            offset = (h - w) / 2
            bbox = (0.0, offset, w, w + offset)

        bbox_list = [torch.tensor([bbox])]
        block_list = [image]

        scale = 1.0
        while True:
            partitions = self._partitioner.partition_image(image)
            if partitions is None:
                break

            bboxes, blocks = partitions
            bbox_list.append(bboxes.to_tensor() * scale)
            block_list.extend(blocks)

            # cannot use `floordiv` which returns floats
            w, h = image.size
            w = int(w / self._rescale)
            h = int(h / self._rescale)
            image = image.resize((w, h))
            scale *= self._rescale

        bboxes_ = BBoxes(torch.cat(bbox_list))
        blocks_ = torch.stack(list(map(self._transforms, block_list)))
        return bboxes_, blocks_

    def __getitem__(self, index: int) -> Batch:
        filename, image, _ = self._getitem(index)
        bboxes, blocks = self._partition(image)
        key = filename.replace('images/', '').replace('.jpg', '').replace('/','-')
        return Batch(id_=key, bboxes=bboxes, blocks=blocks)


@OAKEDatasetRegistry.register_()
class V3DetBlockDataset(BaseBlockDataset):
    pass
from oadp.base.datasets import COCO_17, COCO_48, COCO_48_17


def test_coco() -> None:
    assert len(set(COCO_17)) == 17
    assert len(set(COCO_48)) == 48
    assert len(set(COCO_48_17)) == 65

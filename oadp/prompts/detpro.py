import torch
from mmdet.datasets import LVISV1Dataset


def main() -> None:
    embeddings = torch.load('pretrained/detpro/iou_neg5_ens.pth', 'cpu')

    # lvis annotations have a typo, which is fixed in mmdet
    # we need to change it back, so that the names match
    names: list[str] = list(LVISV1Dataset.CLASSES)
    i = names.index('speaker_(stereo_equipment)')
    names[i] = 'speaker_(stero_equipment)'

    state_dict = dict(
        embeddings=embeddings,
        names=names,
    )
    torch.save(state_dict, 'data/prompts/detpro_lvis.pth')


if __name__ == '__main__':
    main()

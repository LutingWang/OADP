import torch
from todd import Registry

from ..base import globals_


class ClassifierRegistry(Registry):
    pass


@ClassifierRegistry.register()
class ViLDClassifier:

    def __init__(self, pretrained) -> None:
        self.classname = globals_.lvis.all_
        self.prompts = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = self.prompts['embeddings']
        self.embeddings = self.fliter_embeddings(embeddings)
        self.scaler: float = 1
        self.bias: float = 0

    def fliter_embeddings(self, embeddings) -> torch.Tensor:
        names: list[str] = self.prompts['names']
        name2ind: dict = {name: i for i, name in enumerate(names)}
        inds = [name2ind[name] for name in self.classname]
        return embeddings[inds].half()

    @property
    def infos(self) -> tuple[torch.Tensor, float, float]:
        return self.embeddings, self.scaler, self.bias


@ClassifierRegistry.register()
class MLClassifier(ViLDClassifier):

    def __init__(self, pretrained) -> None:
        self.classname = globals_.coco.all_
        self.prompts = torch.load(pretrained, 'cpu')
        embeddings: torch.Tensor = self.prompts['embeddings']
        self.embeddings = self.fliter_embeddings(embeddings)
        self.scaler: float = self.prompts["scaler"].item()
        self.bias: float = self.prompts["bias"].item()

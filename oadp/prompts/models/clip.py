__all__ = [
    'CLIP',
]

from typing import Iterable

import todd
import todd.tasks.natural_language_processing as nlp
import torch
import torch.nn.functional as F
from todd.models.modules import CLIPText

from ..registries import PromptModelRegistry
from .base import BaseModel


@PromptModelRegistry.register_()
class CLIP(BaseModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = nlp.tokenizers.CLIPTokenizer(
            bpe_path='pretrained/clip/clip_bpe.txt.gz',
        )
        model = CLIPText(out_features=512)
        model.load_pretrained('pretrained/clip/ViT-B-32.pt')
        model.requires_grad_(False)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def forward(
        self,
        texts: Iterable[str],
        batch_size: str | None = None,
    ) -> torch.Tensor:
        tokens = self._tokenizer.encodes(texts, max_length=77)
        if todd.Store.cuda:
            tokens = tokens.cuda()
        if batch_size is None:
            embeddings = self._model(tokens)
        else:
            embeddings = torch.cat([
                self._model(tokens_) for tokens_ in tokens.split(batch_size)
            ])
        embeddings = CLIPText.eos(tokens, embeddings)
        embeddings = F.normalize(embeddings)
        return embeddings

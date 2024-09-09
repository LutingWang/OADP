__all__ = [
    'T5',
]

from typing import Iterable

import torch
from transformers import T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

from ..registries import PromptModelRegistry
from .base import BaseModel


@PromptModelRegistry.register_()
class T5(BaseModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tokenizer = T5Tokenizer.from_pretrained('pretrained/t5/t5-large')
        model = T5EncoderModel.from_pretrained('pretrained/t5/t5-large')
        model = model.requires_grad_(False)
        model = model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def forward(
        self,
        texts: Iterable[str],
        batch_size: str | None = None,
    ) -> torch.Tensor:
        assert batch_size is None
        embeddings: list[torch.Tensor] = []
        for text in texts:
            tokens = self._tokenizer(text, return_tensors='pt')
            outputs: BaseModelOutput = self._model(**tokens)
            embedding = outputs.last_hidden_state.mean(0)
            embeddings.append(embedding)
        return torch.stack(embeddings)

__all__ = [
    'T5',
]

from typing import Iterable

import todd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput
from todd.registries import InitWeightsMixin

from ..registries import PromptModelRegistry
from .base import BaseModel


@PromptModelRegistry.register_()
class T5(InitWeightsMixin, BaseModel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tokenizer: T5Tokenizer = AutoTokenizer.from_pretrained(
            'pretrained/t5/t5-large',
        )
        model = T5EncoderModel.from_pretrained('pretrained/t5/t5-large')
        model = model.requires_grad_(False)
        model = model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def init_weights(self, config: todd.Config) -> bool:
        super().init_weights(config)
        return False

    def forward(
        self,
        texts: Iterable[str],
        batch_size: str | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        for text in texts:
            tokens = self._tokenizer(text, return_tensors='pt')
            if todd.Store.cuda:
                tokens = tokens.to('cuda')
            outputs: BaseModelOutput = self._model(**tokens)
            embedding = outputs.last_hidden_state.mean(1)
            embeddings.append(embedding)
        embedding = torch.cat(embeddings)
        return F.normalize(embedding)

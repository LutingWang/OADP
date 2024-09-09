__all__ = [
    'LVISPrompter',
]

import json
import re
from typing import Any

from ..registries import PrompterRegistry
from ..utils import RAM, ViLD, WordNet
from .base import BasePrompter


@PrompterRegistry.register_()
class LVISPrompter(BasePrompter):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ram = RAM()
        self._vild = ViLD()
        self._wordnet = WordNet()

    def load(self) -> list[dict[str, Any]]:
        with open('data/lvis/annotations/lvis_v1_val.json') as f:
            annotations = json.load(f)
        return annotations['categories']

    def _prompt(self, category: dict[str, Any]) -> dict[str, Any]:
        synset = self._wordnet.synset(category['synset'])
        if synset is None:
            definition = category['def']
            synonyms = category['synonyms']
        else:
            definition, synonyms = synset

        descriptions = self._ram(
            re.sub(r'\(.*\)', '', synonym) for synonym in synonyms
        )

        return dict(
            definition=definition,
            synonyms=synonyms,
            descriptions=descriptions,
            definition_encoding=self._model([definition]),
            synonym_encoding=self._model(self._vild(synonyms)),
            description_encoding=(
                None if len(descriptions) == 0 else
                self._model(descriptions, batch_size=128)
            ),
        )

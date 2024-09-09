__all__ = [
    'RAM',
]

import json
import pathlib
import re
from collections import defaultdict
from typing import Iterable


class RAM:

    def __init__(self) -> None:
        descriptions: defaultdict[str, list[str]] = defaultdict(list)

        path = pathlib.Path(__file__).parent

        entry: dict[str, list[str]]

        ram = path / 'ram_tag_list_4585_llm_tag_descriptions.json'
        with ram.open() as f:
            for entry in json.load(f):
                assert len(entry) == 1
                word, description = entry.popitem()
                descriptions[self._normalize(word)].extend(description)

        ram = path / 'openimages_rare_200_llm_tag_descriptions.json'
        with ram.open() as f:
            for entry in json.load(f):
                assert len(entry) == 1
                word, description = entry.popitem()
                descriptions[self._normalize(word)].extend(description)

        self._descriptions = descriptions

    def _normalize(self, word: str) -> str:
        return re.sub(r'[^0-9a-zA-Z]+', '', word).lower()

    def _describe(self, word: str) -> list[str]:
        descriptions = self._descriptions.get(word, [])
        for description in list(descriptions):
            descriptions.extend(description.split('. '))
        return descriptions

    def __call__(self, synonyms: Iterable[str]) -> list[str]:
        synonyms = map(self._normalize, synonyms)
        return sum(map(self._describe, synonyms), [])

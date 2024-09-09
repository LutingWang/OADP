__all__ = [
    'ViLD',
]

import json
import pathlib
from typing import Iterable


class ViLD:

    def __init__(self) -> None:
        path = pathlib.Path(__file__).parent
        templates_path = path / 'templates.json'
        with templates_path.open() as f:
            templates: list[str] = json.load(f)
        self._templates = templates

    def __call__(self, synonyms: Iterable[str]) -> list[str]:
        return [
            template.replace('<|category|>', synonym)
            for template in self._templates
            for synonym in synonyms
        ]

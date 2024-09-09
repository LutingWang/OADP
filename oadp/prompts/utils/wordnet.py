__all__ = [
    'WordNet',
]

from typing import cast

import nltk
from nltk.corpus import wordnet
from nltk.corpus.reader import Synset, WordNetError

nltk.data.path.append('data/wordnet')


class WordNet:

    def synset(self, name: str) -> tuple[str, list[str]] | None:
        try:
            synset: Synset = wordnet.synset(name)
        except WordNetError:
            return None

        definition = synset.definition()
        synonyms = [
            cast(str, synonym).replace('_', ' ')
            for synonym in synset.lemma_names()
        ]
        return definition, synonyms

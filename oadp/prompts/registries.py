__all__ = [
    'PromptModelRegistry',
    'PrompterRegistry',
]

from todd.registries import ModelRegistry

from ..registries import PromptRegistry


class PromptModelRegistry(PromptRegistry, ModelRegistry):
    pass


class PrompterRegistry(PromptRegistry):
    pass

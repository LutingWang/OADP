import torch

class Globals:
    """Entry point for global variables.

    Not to be confused with the global distillation branch.
    """
    sample_num: int
    texts: list[list[str]]
import torch
from omegaconf import OmegaConf

from .base import BaseSelector


class FixedSelector(BaseSelector):
    def __init__(self, vecs: list, cfg: OmegaConf):
        super().__init__(vecs, cfg)
        self.target = cfg.target

    def select_new(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Randomly select a target for the given input spectrogram.
        """
        return torch.ones((spec.shape[0]), dtype=torch.int64) * self.target

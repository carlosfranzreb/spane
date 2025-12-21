import torch
from torch import Tensor
from omegaconf import DictConfig

from .base import BaseSelector


class FixedSelector(BaseSelector):
    def __init__(self, vecs: list, cfg: DictConfig):
        """
        The target defined in the config is selected for all sources.
        """
        super().__init__(vecs, cfg)
        self.target = cfg.target

    def select_new(self, indices: Tensor, batch: list[Tensor]) -> Tensor:
        device = batch[self.config.input.source_is_male].device
        source_is_male = batch[self.config.input.source_is_male].to("cpu")
        source_is_male = source_is_male[indices]
        return (
            torch.ones((source_is_male.shape[0]), dtype=torch.int64, device=device)
            * self.target
        )

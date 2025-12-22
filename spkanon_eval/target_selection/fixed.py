import torch
from torch import Tensor
from omegaconf import DictConfig

from .base import BaseSelector


class FixedSelector(BaseSelector):
    def __init__(self, cfg: DictConfig, target_df: str):
        """The target defined in the config is selected for all sources."""
        super().__init__(cfg, target_df)
        self.target = cfg.target

    def select_new(self, mask: Tensor, batch: dict) -> Tensor:
        device = batch["source"].device
        source = batch["source"].to("cpu")
        source = source[mask]
        return (
            torch.ones((source.shape[0]), dtype=torch.int64, device=device)
            * self.target
        )

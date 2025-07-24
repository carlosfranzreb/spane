"""
Returns the same spectrograms it receives. Used for testing purposes.
"""

import importlib
import torch
from omegaconf import DictConfig

from spkanon_eval.component_definitions import InferComponent


class DummyConverter(InferComponent):
    def __init__(self, config: DictConfig, device: str, **kwargs) -> None:
        """
        Store where the spectrogram, source and target is stored in the batch.
        """
        self.input_spec = config.input.spectrogram
        self.input_len = config.input.n_frames
        self.input_source = config.input.source
        self.input_target = config.input.target
        self.device = device
        self.model = torch.empty(1)
        self.n_targets = config.n_targets
        self.target_selection = None

    def run(self, batch: list) -> list:
        """
        Return the given spectrograms and the targets.
        """
        spec = batch[self.input_spec]
        n_frames = batch[self.input_len]
        source = batch[self.input_source]
        source_is_male = torch.randint(
            2, (source.shape[0],), device=self.device, dtype=torch.bool
        )
        mock_input = torch.zeros(spec.shape[0], dtype=torch.int64, device=self.device)
        target = self.target_selection.select(mock_input, source, source_is_male)
        return {"spectrogram": spec, "n_frames": n_frames, "target": target}

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)

    def init_target_selection(self, cfg: DictConfig, *args) -> None:
        """
        Initialize the target selection.
        """

        targets = torch.arange(self.n_targets).to(self.device)
        module_str, cls_str = cfg.cls.rsplit(".", 1)
        module = importlib.import_module(module_str)
        cls = getattr(module, cls_str)
        self.target_selection = cls(targets, cfg, *args)

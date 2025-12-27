"""
Synthesizer that does nothing, to be used in testing.
"""

import torch

from spkanon_eval.component_definitions import InferComponent


class DummySynthesizer(InferComponent):
    def __init__(self, config, device):
        """
        Store where the spectrogram is stored in the batch.
        """
        self.spec_input = config.input.spectrogram
        self.input_len = config.input.n_frames
        self.device = device
        self.model = torch.empty(1)
        self.upsampling_rate = 512

    def run(self, batch: dict) -> torch.Tensor:
        """
        Return a tensor of random values of shape (batch_size, 1, spec_frames * 512).
        """
        specs = batch[self.spec_input]
        n_frames = batch[self.input_len]
        length = specs.shape[2] * 512
        converted = torch.randn((specs.shape[0], 1, length), device=self.device)
        n_samples = n_frames * self.upsampling_rate
        return converted, n_samples

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(device)

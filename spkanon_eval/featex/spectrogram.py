"""
Compute the spectrogram of the audio tensor after normalizing it.
"""

import logging
import torch
from torchaudio.transforms import MelSpectrogram

from spkanon_eval.component_definitions import InferComponent
from spkanon_eval.datamodules import AudioBatch

LOGGER = logging.getLogger("progress")


class SpecExtractor(InferComponent):
    def __init__(self, config, device):
        self.config = config
        self.mel_trafo = MelSpectrogram(
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            win_length=config.win_length,
            hop_length=config.hop_length,
        )
        self.mel_trafo.to(device)

        LOGGER.info("Initialized mel-spectrogram extractor with config:")
        LOGGER.info(f"\tn_mels: {config.n_mels}")
        LOGGER.info(f"\tn_fft: {config.n_fft}")
        LOGGER.info(f"\twin_length: {config.win_length}")
        LOGGER.info(f"\thop_length: {config.hop_length}")

    def run(self, batch: AudioBatch) -> dict:
        """
        Computes the mel-spectrograms of the waveforms after normalizing them.

        Args:
            batch: a list with a tensor comprising waveforms in first position, and the
            number of samples per item in the batch in third position.

        Returns:
            a dictionary with the output of the specified WavLM layer under the key "feats" and
            the number of feats for each sample under "n_feats".
        """
        audio_norm = batch.audios / torch.max(torch.abs(batch.audios))
        melspecs = self.mel_trafo(audio_norm)
        lengths = torch.ceil(batch.lens / self.config.hop_length).to(torch.long)
        return {"spectrogram": melspecs, "n_frames": lengths}

    def to(self, device):
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.mel_trafo.to(device)

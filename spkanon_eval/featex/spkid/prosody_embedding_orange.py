import parselmouth
import numpy as np
import logging
import torch
from omegaconf import OmegaConf

from spkanon_eval.evaluate import SAMPLE_RATE
from spkanon_eval.component_definitions import InferComponent

from .speaker_wavlm_pro import EmbeddingsModel

LOGGER = logging.getLogger("progress")


class ProsodyEmbeddingOrange(InferComponent):

    def __init__(self, config: OmegaConf, device: str):
        self.emb_model = EmbeddingsModel.from_pretrained("Orange/Speaker-wavLM-pro")
        self.emb_model.eval()
        self.emb_model.to(device)
        self.device = device

    def to(self, device: str):
        self.device = device

    @torch.inference_mode()
    def run(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Return speaker embeddings for the given batch of utterances.
        ! The max. duration is 20 seconds. Longer utterances are trimmed.

        Args:
            batch: A list of the waveforms in first position

        Returns:
            A tensor containing the prosody embeddings with shape
            (batch_size, embedding_dim).
        
        """
        return self.emb_model(batch[0].to(self.device)[:, :int(20 * SAMPLE_RATE)])

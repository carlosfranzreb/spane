import unittest

import torch
import torchaudio
from omegaconf import OmegaConf

from spkanon_eval.featex import ProsodyEmbedding


class TestProsodyEmbedding(unittest.TestCase):
    def setUp(self):
        """Set up test configuration and mock audio data."""
        # Create test configuration
        self.config = OmegaConf.load(
            "spkanon_eval/config/components/asv/spkid/prosody.yaml"
        )["spkid"]

        self.device = "cpu"
        self.prosody_model = ProsodyEmbedding(self.config, self.device)

        # Create a batch
        audio = torchaudio.load(
            "spkanon_eval/tests/data/LibriSpeech/dev-clean-2/1988/24833/1988-24833-0000.flac"
        )[0].squeeze()
        audios = torch.stack([audio, audio], dim=0)
        self.batch = [
            audios,
            torch.tensor([audios.shape[1], audios.shape[1] - 16000], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
        ]

    def test_segmentations(self):
        """
        Test the two segmentations: that their outputs are finite non-zero
        embeddings and that they differ from one another.
        """
        embeddings = dict()
        for method in ["relative", "pitch"]:
            self.prosody_model.segmentation_method = method
            if method == "relative":
                self.prosody_model.feature_segments = {
                    "f0": "all",
                    "energy": "all",
                }

            with torch.inference_mode():
                embs = self.prosody_model.run(self.batch)

            # Check output type and device
            self.assertIsInstance(embs, torch.Tensor)
            self.assertEqual(embs.device, self.batch[0].device)
            self.assertEqual(embs.dtype, torch.float32)

            # Check that embs are finite, not all zeros and correct dim
            self.assertTrue(torch.all(torch.isfinite(embs)))
            self.assertFalse(torch.allclose(embs, torch.zeros_like(embs)))
            self.assertEqual(embs.shape[0], len(self.batch[0]))
            embeddings[method] = embs

        # Test that the segmentations produce embeddings of different shapes
        self.assertNotEqual(embeddings["pitch"].shape, embeddings["relative"].shape)

    def test_device_switching(self):
        """Test that device switching works correctly."""
        original_device = self.prosody_model.device
        new_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prosody_model.to(new_device)
        self.assertEqual(self.prosody_model.device, new_device)

        self.prosody_model.to(original_device)
        self.assertEqual(self.prosody_model.device, original_device)

    def test_config_parameters_used(self):
        """Test that configuration parameters are properly used."""
        config_modified = self.config.copy()
        config_modified.pitch_floor = 50.0
        model_modified = ProsodyEmbedding(config_modified, self.device)

        with torch.inference_mode():
            embeddings_original = self.prosody_model.run(self.batch)
            embeddings_modified = model_modified.run(self.batch)

        # Should produce different results
        if embeddings_original.shape == embeddings_modified.shape:
            self.assertFalse(
                torch.allclose(embeddings_original, embeddings_modified, atol=1e-4)
            )

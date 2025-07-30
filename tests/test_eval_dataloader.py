import unittest
import json

from omegaconf import OmegaConf
import torchaudio
import torch

from spkanon_eval.datamodules import eval_dataloader
from spkanon_eval.featex import Whisper


class TestEvalDataloader(unittest.TestCase):
    def setUp(self):
        """
        - Create a test directory and add a logger there; the spk2id mapping will be
            dumped there.
        - Get the test datafiles.
        """
        self.datafile = "spane/tests/datafiles/ls-dev-clean-2.txt"
        self.device = "cpu"
        self.config = OmegaConf.load("spane/config/datasets/config.yaml")["config"]
        self.config.sample_rate = 16000
        self.config.sample_rate_out = 16000
        self.config.sample_rate_in = 24000

        whisper_config = OmegaConf.load(
            "spane/config/components/asr/whisper_tiny.yaml"
        )["whisper_tiny"]
        self.model = Whisper(whisper_config, self.device)

    def test_eval_dataloader(self):
        """
        Test that the eval dataloader returns the correct number of batches and that
        the content of the batches matches that of the test datafiles. This test
        also checks that the resampling works.

        We asume that the dataset's spk2id mapping is correct.
        """
        dl = eval_dataloader(self.config, self.datafile, self.model)
        samples = open(self.datafile).readlines()

        for batch, data in dl:
            batch_size = batch[0].shape[0]
            objs = list()
            for _ in range(batch_size):
                objs.append(json.loads(samples.pop(0)))

            self.assertEqual(len(batch), 3)  # audio, speakers, lengths
            for i in range(batch_size):
                obj = objs[i]
                audio_true, sr = torchaudio.load(obj["path"])
                sample_data = data[i]
                if sr != self.config.sample_rate_in:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.config.sample_rate_in
                    )
                    audio_true = resampler(audio_true)
                self.assertTrue(audio_true.shape[1] <= batch[0][i].shape[0])
                self.assertTrue(
                    torch.allclose(audio_true, batch[0][i, : audio_true.shape[1]])
                )
                self.assertTrue(torch.sum(batch[0][i, audio_true.shape[1] :]) == 0)
                self.assertEqual(audio_true.shape[1], batch[2][i])

                # check metadata
                for key in obj.keys():
                    if key == "path":
                        continue
                    a, b = obj[key], sample_data[key]
                    if not isinstance(a, str):
                        a = str(a)
                    if not isinstance(b, str):
                        b = str(b)
                    self.assertEqual(a, b)

        # ensure that all samples have been read
        self.assertEqual(len(samples), 0)

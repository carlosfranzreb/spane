import unittest
import json

import torch
import torchaudio

from spkanon_eval.datamodules import SpeakerIdDataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        """
        - Create a test directory and add a logger there; the spk2id mapping will be
            dumped there.
        - Get the test datafiles and create the dataset.
        """
        self.datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"
        self.sample_rate = 16000
        self.dataset = SpeakerIdDataset(self.datafile, self.sample_rate, {100: 1})

    def test_dataset_length(self):
        n_samples = 0
        with open(self.datafile) as f:
            n_samples += len(f.readlines())
        self.assertEqual(len(self.dataset), n_samples)

    def test_dataset_items(self):
        """
        Ensure that the dataset returns the correct audio, speaker IDs and audio
        durations. We assume that the dataset's spk2id mapping is correct.
        """

        for sample_idx, line in enumerate(open(self.datafile).readlines()):
            obj = json.loads(line)
            audio_true, sr = torchaudio.load(obj["path"])
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                )
                audio_true = resampler(audio_true)

            audio_dataset, spk, n_samples = self.dataset[sample_idx]
            self.assertEqual(audio_true.shape[1], audio_dataset.shape[1])
            self.assertEqual(obj["speaker_id"], spk)
            self.assertTrue(torch.allclose(audio_true, audio_dataset))
            self.assertEqual(n_samples, audio_true.shape[1])

    def test_data_chunking(self):
        """
        Check that the dataset chunks the samples with the given sizes per duration.
        """
        audios, speakers, durations = list(), list(), list()
        for line in open(self.datafile).readlines():
            obj = json.loads(line)
            audios.append(torchaudio.load(obj["path"])[0].squeeze())
            speakers.append(int(obj["speaker_id"]))
            durations.append(int(audios[-1].shape[0]))

        dataset = SpeakerIdDataset(self.datafile, self.sample_rate, {12: 2, 4: 3})
        expected_sizes = [2, 2, 2, 3, 1]
        sample_idx = 0
        for batch_idx, batch in enumerate(dataset):
            self.assertEqual(batch[1].shape[0], expected_sizes[batch_idx])
            for batch_sample_idx in range(expected_sizes[batch_idx]):
                self.assertEqual(batch[1][batch_sample_idx], speakers[sample_idx])
                self.assertEqual(batch[2][batch_sample_idx], durations[sample_idx])
                self.assertTrue(
                    torch.allclose(
                        batch[0][batch_sample_idx, : batch[2][batch_sample_idx]],
                        audios[sample_idx],
                    )
                )
                sample_idx += 1

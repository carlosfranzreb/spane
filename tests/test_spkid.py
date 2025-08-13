"""
Test that the spkid model correctly initializes and runs speechbrain models.
We test with the test samples from common voice on CPU.
"""

import unittest
import os
import shutil
import json
import copy

from omegaconf import OmegaConf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from spkanon_eval.featex import SpkId
from spkanon_eval.featex import SpkIdConcat
from spkanon_eval.datamodules.dataset import load_audio
from spkanon_eval.utils import seed_everything


SAMPLE_RATE = 16000


class TestSpkid(unittest.TestCase):
    def setUp(self):
        """
        Declare the configs for the spkid models and the data directory and seed
        everything.
        """

        seed_everything(42)
        self.cfg = OmegaConf.create(
            {
                "path": "speechbrain/spkrec-xvect-voxceleb",
                "emb_model_ckpt": None,
                "num_workers": 0,
                "train_config": "spane/config/components/asv/spkid/train_xvector_debug.yaml",
                "al_weight": {
                    "n_epochs_zero": 1,
                    "n_epochs_max": 0,
                    "max_weight": 0.0,
                },
            }
        )
        self.data_dir = "spane/tests/data/LibriSpeech/dev-clean-2/1988/24833"

    def get_batch(self, samples: list[str]) -> list[Tensor, Tensor, Tensor]:
        """Return a batch given the samples."""
        audios, speakers = list(), list()
        for sample in samples:
            audios.append(load_audio(os.path.join(self.data_dir, sample), SAMPLE_RATE))
            speakers.append(0)

        return [
            pad_sequence(audios, batch_first=True),
            torch.tensor([audio.shape[0] for audio in audios]),
            torch.tensor(speakers),
        ]

    def test_batches(self):
        """Ensure that batching works with the x-vector model."""

        model = SpkId(self.cfg, "cpu")

        # test with batches of 1 sample
        spk = torch.tensor([0], dtype=torch.int64)
        with self.subTest("1 sample"):
            for audiofile in os.listdir(self.data_dir):
                audio = load_audio(os.path.join(self.data_dir, audiofile), SAMPLE_RATE)
                audio = torch.tensor(audio).unsqueeze(0)
                length = torch.tensor([audio.shape[1]], dtype=torch.int64)

                emb = model.run((audio, spk, length))
                self.assertTrue(emb.shape == (1, 512))

        # test with one batch of 2 samples
        samples = os.listdir(self.data_dir)[:2]
        batch = self.get_batch(samples)

        emb = model.run(batch)
        with self.subTest("2 samples"):
            self.assertTrue(emb.shape == (2, 512))

    def test_concat(self):
        """
        Ensure that the SpkIdConcat class correctly concatenates the outputs of the
        spkid models.
        """

        ecapa_cfg = self.cfg.copy()
        ecapa_cfg.path = "speechbrain/spkrec-ecapa-voxceleb"
        model_cfg = [ecapa_cfg, self.cfg]
        expected_shape = [2, 704]

        concat_cfg = OmegaConf.create(
            {"cls": "spkanon_eval.featex.spkid.SpkIdConcat", "models": model_cfg}
        )
        concat_model = SpkIdConcat(concat_cfg, "cpu")
        samples = os.listdir(self.data_dir)[:2]
        batch = self.get_batch(samples)

        out = concat_model.run(batch)
        self.assertEqual(list(out.shape), expected_shape)

    def test_train(self):
        """Test that the spkid model is trained on the given datafile."""

        exp_folder = "spane/tests/logs/spkid_train"
        if os.path.isdir(exp_folder):
            shutil.rmtree(exp_folder)
        os.makedirs(os.path.join(exp_folder))
        datafile = "spane/tests/datafiles/ls-dev-clean-2.txt"
        n_speakers = 3
        n_targets = 3
        model = SpkId(self.cfg, "cpu")
        old_state_dict = copy.deepcopy(model.model.state_dict())
        model.train(exp_folder, datafile, n_speakers, n_targets)

        # gather the utterances from the datafile
        expected_samples = dict()
        for line in open(os.path.join(datafile)):
            obj = json.loads(line)
            expected_samples[obj["path"]] = obj

        # check the train and val samples
        for split_file, n_lines in zip(["train.csv", "val.csv"], [18, 5]):
            with open(os.path.join(exp_folder, split_file)) as f:
                train_lines = f.readlines()
            self.assertEqual(len(train_lines), n_lines)

        # check that the model is trained one epoch
        log_file = os.path.join(exp_folder, "train_log.txt")
        self.assertTrue(os.path.isfile(log_file))
        with open(log_file) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        new_state_dict = model.model.state_dict()
        self.assertTrue(
            any(
                [
                    not torch.equal(old, new)
                    for old, new in zip(
                        old_state_dict.values(), new_state_dict.values()
                    )
                ]
            )
        )

        # check that the model is saved
        subdir = None
        for sub in os.listdir(exp_folder):
            if sub.startswith("CKPT"):
                subdir = sub
                break
        model_file = os.path.join(exp_folder, subdir, "embedding_model.ckpt")
        self.assertTrue(os.path.isfile(model_file))

        shutil.rmtree(exp_folder)

    def test_al_training(self):
        """
        Test that adversarial training changes the spkid model's weights.

        We train two models, one with and one without adversarial loss, and check
        that their weights are different.
        """
        exp_folder_no_al = "spane/tests/logs/spkid_train_no_al"
        exp_folder_al = "spane/tests/logs/spkid_train_al"
        for folder in [exp_folder_no_al, exp_folder_al]:
            if os.path.isdir(folder):
                shutil.rmtree(folder)

        datafile = "spane/tests/datafiles/ls-dev-clean-2.txt"
        n_speakers = 3
        n_targets = 3

        # Setup configs and initialize models
        cfg_no_al = self.cfg
        cfg_al = copy.deepcopy(self.cfg)
        cfg_al.al_weight.n_epochs_zero = 0
        cfg_al.al_weight.n_epochs_max = 1
        cfg_al.al_weight.max_weight = 1.0

        model_no_al = SpkId(cfg_no_al, "cpu")
        model_al = SpkId(cfg_al, "cpu")

        # Check that they start with the same weights
        for p1, p2 in zip(
            model_no_al.model.mods.embedding_model.state_dict().values(),
            model_al.model.mods.embedding_model.state_dict().values(),
        ):
            self.assertTrue(torch.equal(p1, p2))

        # Train models
        model_no_al.train(exp_folder_no_al, datafile, n_speakers, n_targets)
        model_al.train(exp_folder_al, datafile, n_speakers, n_targets)

        # The weights of the two models should now be different.
        state_dict_no_al = model_no_al.model.mods.embedding_model.state_dict()
        state_dict_al = model_al.model.mods.embedding_model.state_dict()
        are_different = any(
            not torch.equal(p1, p2)
            for p1, p2 in zip(state_dict_no_al.values(), state_dict_al.values())
        )
        self.assertTrue(
            are_different,
            "Model weights should be different after AL training.",
        )

        for folder in [exp_folder_no_al, exp_folder_al]:
            shutil.rmtree(folder)

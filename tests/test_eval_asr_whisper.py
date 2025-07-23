"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""

import os
import shutil
import unittest
import json

from omegaconf import OmegaConf

from spkanon_eval.featex import Whisper


class TestEvalWhisper(unittest.TestCase):
    def setUp(self):
        """
        Create the evaluation component and run it with the LibriSpeech dev-clean data.
        """

        self.exp_folder = "spkanon_eval/tests/logs/asr_whisper"
        if os.path.isdir(self.exp_folder):
            shutil.rmtree(self.exp_folder)
        os.makedirs(os.path.join(self.exp_folder))
        self.datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"

        # config = OmegaConf.create(
        #     {
        #         "train": False,
        #         "size": "tiny",
        #         "output": "text",
        #         "batch_size": 4,
        #         "data": {
        #             "config": {
        #                 "sample_rate_in": 16000,
        #                 "num_workers": 0,
        #                 "chunk_sizes": {"ls-dev-clean-2": {100: 1}},
        #             },
        #         },
        #     }
        # )
        config = OmegaConf.load("spkanon_eval/config/components/asr/whisper_tiny.yaml")[
            "whisper_tiny"
        ]
        config.data = OmegaConf.load("spkanon_eval/config/datasets/config.yaml")
        config.data.config.sample_rate = 16000
        config.data.config.sample_rate_out = 16000
        config.data.config.sample_rate_in = 16000
        self.whisper = Whisper(config, "cpu")
        self.whisper.eval_dir(self.exp_folder, self.datafile)
        self.results_dir = os.path.join(self.exp_folder, "eval", "whisper-tiny")
        self.wers = list()

    def tearDown(self) -> None:
        """Remove the experiment folder."""
        shutil.rmtree(self.exp_folder)

    def test_sample_results(self):
        """Check the content of the `ls-dev-clean-2.txt` file."""

        with open(os.path.join(self.results_dir, "ls-dev-clean-2.txt")) as f:
            results = [line.split() for line in f.readlines()]
        results = results[1:]

        with open(self.datafile) as f:
            expected = [json.loads(line) for line in f.readlines()]

        self.assertEqual(len(results), len(expected))

        for obj in expected:
            out_obj = None
            for out in results:
                if out[0] == obj["path"]:
                    out_obj = out
                    break
            self.assertTrue(out_obj is not None)
            self.assertEqual(len(obj["text"].split()), int(out_obj[2]))
            wer = float(out_obj[3])
            self.assertGreaterEqual(wer, 0)
            self.assertLessEqual(wer, 0.3)
            self.wers.append(wer)

    def test_analysis(self):
        """
        Test the analysis output:

        - `all.txt` should contain the avg. WER of all samples.
        - `gender.txt` should contain the avg. WER per gender.
        """

        # ensure that the correct number of files is created
        self.assertEqual(len(os.listdir(self.results_dir)), 3)

        # if the analysis has not been run, run it
        if len(self.wers) == 0:
            self.test_sample_results()

        # check the content of the `all.txt` file
        with open(os.path.join(self.results_dir, "all.txt")) as f:
            all_results = f.readlines()

        self.assertEqual(len(all_results), 2)
        computed_wer = float(all_results[1].split()[-1])
        self.assertAlmostEqual(computed_wer, sum(self.wers) / len(self.wers), places=2)

        # check the content of the `gender.txt` file
        with open(self.datafile) as f:
            sample_genders = [json.loads(line)["gender"] for line in f]

        with open(os.path.join(self.results_dir, "gender.txt")) as f:
            gender_results = [line.split() for line in f.readlines()]

        self.assertEqual(len(gender_results) - 1, len(set(sample_genders)))

        for line in gender_results[1:]:
            gender = line[1]
            computed_wer = float(line[-1])
            true_wers = [
                self.wers[idx] * (sample_genders[idx] == gender)
                for idx in range(len(self.wers))
            ]
            true_wer = sum(true_wers) / sum([g == gender for g in sample_genders])
            self.assertAlmostEqual(computed_wer, true_wer, places=1)

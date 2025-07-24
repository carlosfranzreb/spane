"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""

import json
import os
import shutil
import unittest

from omegaconf import OmegaConf

from spkanon_eval.evaluation import NisqaEvaluator


class TestEvalNisqa(unittest.TestCase):
    def setUp(self) -> None:
        """Run NISQA with the LibriSpeech dev-clean data."""

        self.exp_folder = "spkanon_eval/tests/logs/naturalness_nisqa"
        if os.path.isdir(self.exp_folder):
            shutil.rmtree(self.exp_folder)
        os.makedirs(os.path.join(self.exp_folder))
        self.datafile = "spkanon_eval/tests/datafiles/ls-dev-clean-2.txt"

        nisqa_config = OmegaConf.load(
            "spkanon_eval/config/components/naturalness/nisqa.yaml"
        )["naturalness_nisqa"]
        nisqa_config.num_workers = 0
        self.nisqa = NisqaEvaluator(nisqa_config, "cpu")
        self.nisqa.eval_dir(self.exp_folder, self.datafile)
        self.results_dir = os.path.join(self.exp_folder, "eval", "nisqa")

    def tearDown(self) -> None:
        """Remove the experiment folder."""
        shutil.rmtree(self.exp_folder)

    def test_nisqa(self):
        """
        Check that NISQA produces valid scores for the utterances in the datafile. As
        we are using original data, we expect the MOS scores to be between 4 and 5.
        """

        # assert that the directory exists
        self.assertTrue(os.path.isdir(self.results_dir))

        # gather the utterances from the datafile
        expected_utts = list()
        for line in open(os.path.join(self.datafile)):
            expected_utts.append(json.loads(line)["path"])

        # gather the utts from the NISQA output and ensure the scores are valid
        output_file = os.path.join(self.results_dir, "ls-dev-clean-2.txt")
        eval_utts = list()
        with open(output_file) as f:
            next(f)  # skip header
            for line in f:
                utt, score = line.split()
                self.assertGreaterEqual(float(score), 4)
                self.assertLessEqual(float(score), 5)
                eval_utts.append(utt)

        # assert that all utts are included in the output
        self.assertCountEqual(expected_utts, eval_utts)

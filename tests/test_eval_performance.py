"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""

import os
import unittest
import shutil
import sys

from omegaconf import OmegaConf
import torch

from spkanon_eval.evaluation import PerformanceEvaluator


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 1)
        self.device = "cpu"

    def forward(self, *args):
        input = torch.tensor([1.0])
        return self.fc(input)


class TestEvalPerformance(unittest.TestCase):
    def test_results(self):
        """
        Test whether the Performance results match the expected values. We only test
        the CPU results, not the GPU results.
        """

        # create/empty experiment folder
        exp_folder = "spkanon_eval/tests/logs/performance"
        if os.path.isdir(exp_folder):
            shutil.rmtree(exp_folder)
        os.makedirs(os.path.join(exp_folder))

        self.config = OmegaConf.load(
            "spkanon_eval/config/components/performance/performance_20s.yaml"
        )["performance"]
        self.config.data = OmegaConf.load("spkanon_eval/config/datasets/config.yaml")
        self.config.data.config.sample_rate = 16000
        self.config.data.config.sample_rate_out = 16000
        self.config.data.config.sample_rate_in = 16000

        evaluator = PerformanceEvaluator(self.config, "cpu", DummyModel())
        evaluator.eval_dir(exp_folder)
        results_dir = os.path.join(exp_folder, "eval", "performance")

        # assert that both directories contain the correct number of files
        self.assertTrue(os.path.isdir(results_dir))
        self.assertEqual(len(os.listdir(results_dir)), 2)

        # assert that the results files contain the same lines
        for fname in os.listdir(results_dir):
            with open(os.path.join(results_dir, fname)) as f:
                results = f.readlines()

            # for `cpu_specs.txt`, compare it with the CPU in this machine
            if fname == "cpu_specs.txt":
                f_expected = os.path.join(results_dir, fname)
                os = sys.platform
                if os == "darwin":
                    os.system(f"sysctl -a | grep machdep.cpu > {f_expected}")
                elif os == "linux":
                    os.system(f"lscpu > {f_expected}")
                else:
                    raise NotImplementedError("Unsupported operating system.")
                with open(os.path.join(results_dir, fname)) as f:
                    expected = f.readlines()
                with self.subTest(fname=fname):
                    self.assertEqual(results, expected)

            # check that the header and first col match, ignore the numbers
            else:
                with open(os.path.join(results_dir, fname)) as f:
                    expected = f.readlines()
                with self.subTest(fname=fname):
                    self.assertEqual(results[0], expected[0])
                    self.assertEqual(
                        [line.split()[0] for line in results],
                        [line.split()[0] for line in expected],
                    )

        shutil.rmtree(exp_folder)

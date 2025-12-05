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
        self.device = "cpu"
        self.fc = torch.nn.Linear(1, 1)

    def forward(self, *args):
        self.fc.to(self.device)
        x = torch.tensor([1.0], device=self.device)
        return self.fc(x)


class TestEvalPerformance(unittest.TestCase):
    def test_results(self):
        """
        Test whether the Performance results match the expected values. We only test
        the CPU results, not the GPU results.
        """

        # create/empty experiment folder
        exp_folder = "spane/tests/logs/performance"
        if os.path.isdir(exp_folder):
            shutil.rmtree(exp_folder)
        os.makedirs(os.path.join(exp_folder))

        self.config = OmegaConf.load(
            "spane/config/components/performance/performance_20s.yaml"
        )["performance"]
        self.config.data = OmegaConf.load("spane/config/datasets/config.yaml")
        self.config.data.config.sample_rate = 16000
        self.config.data.config.sample_rate_out = 16000
        self.config.data.config.sample_rate_in = 16000

        evaluator = PerformanceEvaluator(self.config, "cpu", DummyModel())
        evaluator.eval_dir(exp_folder)
        results_dir = os.path.join(exp_folder, "eval", "performance")

        # assert that both directories contain the correct number of files
        n_files = 4 if torch.cuda.is_available else 2
        self.assertTrue(os.path.isdir(results_dir))
        self.assertEqual(len(os.listdir(results_dir)), n_files)

        # assert that the results files contain the same lines
        for fname in os.listdir(results_dir):
            with open(os.path.join(results_dir, fname)) as f:
                results = f.readlines()

            # for `cpu_specs.txt`, compare it with the CPU in this machine
            if fname == "cpu_specs.txt":
                f_expected = os.path.join(
                    results_dir, fname + ".expected"
                )  # Use distinct name to avoid overwrite issues
                operating_system = sys.platform
                if operating_system == "darwin":
                    os.system(f"sysctl -a | grep machdep.cpu > {f_expected}")
                elif operating_system == "linux":
                    os.system(f"lscpu > {f_expected}")
                else:
                    raise NotImplementedError("Unsupported operating system.")

                with open(os.path.join(results_dir, fname)) as f:
                    results = f.readlines()
                with open(f_expected) as f:
                    expected = f.readlines()

                def filter_volatile(lines):
                    """Ignore specs whose values change over time."""
                    volatile_keys = ["CPU(s) scaling MHz", "CPU MHz", "BogoMIPS"]
                    out = [
                        line
                        for line in lines
                        if not any(key in line for key in volatile_keys)
                    ]
                    return out

                with self.subTest(fname=fname):
                    self.assertEqual(
                        filter_volatile(results), filter_volatile(expected)
                    )

            # check that the header and first col match, ignore the numbers
            elif fname == "cpu_inference.txt":
                with open(os.path.join(results_dir, fname)) as f:
                    expected = f.readlines()
                with self.subTest(fname=fname):
                    self.assertEqual(results[0], expected[0])
                    self.assertEqual(
                        [line.split()[0] for line in results],
                        [line.split()[0] for line in expected],
                    )

        shutil.rmtree(exp_folder)

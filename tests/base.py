"""
This base class should be inherited by all test classes.
- It provides a few helper methods and a setUp and tearDown that start
    and experiment and delete it at the end.
"""

import unittest
from argparse import ArgumentParser
from tempfile import NamedTemporaryFile

from omegaconf import OmegaConf

from run import setup
from spkanon_eval.main import main


MARKER_FILE = ".TEST-FOLDER"
LOG_DIR = "spkanon_eval/tests/logs"


class BaseTestClass(unittest.TestCase):
    def setUp(self):
        """
        Create the config object with the StarGANv2-VC pipeline. Training is disabled,
        inference and evaluation are enabled.
        It will be modified by the test class before running the experiment with
        `run_pipeline()`. The logging directory will be stored in `self.log_dir`, and
        deleted after the test with `tearDown()`.
        """

        self.init_config = OmegaConf.create(
            {
                "name": "Test",
                "inference": {
                    "run": True,
                    "consistent_targets": False,
                    "gender_conversion": None,
                    "input": {"spectrogram": "spectrogram", "target": "target"},
                },
                "seed": 0,
                "log_dir": LOG_DIR,
                "device": "cpu",
                "sample_rate": 24000,
                "batch_size": 2,
                "target_selection_cfg": "spkanon_eval/config/components/target_selection/random.yaml",
                "featex": {
                    "spectrogram": {
                        "cls": "spkanon_eval.featex.spectrogram.SpecExtractor",
                        "n_mels": 80,
                        "n_fft": 2048,
                        "win_length": 1200,
                        "hop_length": 300,
                    }
                },
                "featproc": {
                    "dummy": {
                        "cls": "spkanon_eval.featproc.dummy.DummyConverter",
                        "input": {
                            "spectrogram": "spectrogram",
                            "n_frames": "n_frames",
                            "source": "source",
                            "target": "target",
                        },
                        "n_targets": 10,
                    },
                    "output": {
                        "featex": [],
                        "featproc": ["spectrogram", "n_frames", "target"],
                    },
                },
                "synthesis": {
                    "cls": "spkanon_eval.synthesis.dummy.DummySynthesizer",
                    "sample_rate": "${sample_rate}",
                    "input": {"spectrogram": "spectrogram", "n_frames": "n_frames"},
                },
                "data": {
                    "config": {
                        "root_folder": "spkanon_eval/tests/data",
                        "sample_rate": "${sample_rate}",
                        "batch_size": "${batch_size}",
                    },
                    "datasets": {
                        "eval": [
                            "spkanon_eval/data/debug/ls-dev-clean-2.txt",
                        ],
                        "train_eval": ["spkanon_eval/data/debug/ls-dev-clean-2.txt"],
                    },
                },
                "eval": {
                    "config": {
                        "baseline": False,
                        "exp_folder": None,
                        "sample_rate": "${synthesis.sample_rate}",
                    },
                    "components": dict(),  # will be filled by the test class
                },
            }
        )


def run_pipeline(config):
    """
    Run the pipeline with the current config.
    """
    args = ArgumentParser()
    args.device = config.device
    args.num_workers = 0

    # save the config to a file and pass the file to the setup function
    with NamedTemporaryFile(mode="w+", encoding="utf-8") as tmp_file:
        OmegaConf.save(config, tmp_file.name)
        args.config = tmp_file.name
        config, log_dir = setup(args)

    main(config, log_dir)
    return config, log_dir

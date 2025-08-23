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
LOG_DIR = "spane/tests/logs"


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
                "sample_rate_in": 16000,
                "sample_rate_out": 16000,
                "target_selection_cfg": "spane/config/components/target_selection/random.yaml",
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
                        "n_targets": 20,
                    },
                    "output": {
                        "featex": [],
                        "featproc": ["spectrogram", "n_frames", "target"],
                    },
                },
                "synthesis": {
                    "cls": "spkanon_eval.synthesis.dummy.DummySynthesizer",
                    "sample_rate_in": "${sample_rate_in}",
                    "sample_rate_out": "${sample_rate_out}",
                    "input": {"spectrogram": "spectrogram", "n_frames": "n_frames"},
                },
                "data": {
                    "config": {
                        "root_folder": "spane/tests/data",
                        "sample_rate": 16000,
                        "sample_rate_in": "${sample_rate_in}",
                        "sample_rate_out": "${sample_rate_out}",
                    },
                    "datasets": {
                        "eval": [
                            "spane/data/debug/ls-dev-clean-2.txt",
                        ],
                        "train_eval": ["spane/data/debug/ls-dev-clean-2.txt"],
                        "asv": {
                            "trials": None,
                            "enrolls": None,
                        },
                    },
                },
                "eval": {
                    "config": {
                        "baseline": False,
                        "exp_folder": None,
                        "sample_rate_in": "${synthesis.sample_rate_out}",
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
    args.config_overrides = dict()

    # save the config to a file and pass the file to the setup function
    with NamedTemporaryFile(mode="w+", encoding="utf-8") as tmp_file:
        OmegaConf.save(config, tmp_file.name)
        args.config = tmp_file.name
        config = setup(args)

    main(config)
    return config

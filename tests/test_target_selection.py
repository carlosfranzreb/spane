"""
Test whether the targets are consistent across the utterances of each speaker.
The base target selector is reponsible for this.
"""

import unittest
import torch
from omegaconf import OmegaConf

from spkanon_eval.target_selection import RandomSelector


TARGET_DF = "spane/tests/datafiles/ls-dev-clean-2.txt"


class TestTargetSelection(unittest.TestCase):
    def test_consistent_selection(self):
        """
        Ensure that targets remain or change between calls, depending on the config.
        """
        for consistent_targets in [True, False]:
            cfg = OmegaConf.create({"consistent_targets": consistent_targets})
            selector = RandomSelector(cfg, TARGET_DF)
            batch = {
                "feats": torch.randn(6, 10),
                "source": torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64),
            }
            target_1 = selector.select(batch)
            target_2 = selector.select(batch)

            all_same = torch.all(target_1 == target_2)
            if consistent_targets:
                self.assertTrue(all_same)
                self.assertTrue(target_1[0] == target_1[1])
                self.assertTrue(target_1[2] == target_1[3] == target_1[4])
            else:
                self.assertFalse(all_same)

    def test_same_source_target(self):
        """
        Check that the same ID is avoided when the dataset is the same.
        """
        cfg = OmegaConf.create(
            {"consistent_targets": False, "same_source_target": True}
        )
        selector = RandomSelector(cfg, TARGET_DF)
        batch = {
            "feats": torch.randn(6, 10),
            "source": torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64),
        }

        for _ in range(5):
            targets = selector.select(batch)
            self.assertTrue(torch.all(batch["source"] != targets))

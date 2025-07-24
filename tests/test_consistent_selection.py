"""
Test whether the targets are consistent across the utterances of each speaker.
The base target selector is reponsible for this.
"""

import unittest
import torch
from omegaconf import OmegaConf

from spkanon_eval.target_selection import RandomSelector


class TestConsistentSelection(unittest.TestCase):
    def test_consistent_selection(self):
        # ensure that targets remain or change between calls, depending on the config
        for consistent_targets in [True, False]:
            cfg = OmegaConf.create({"consistent_targets": consistent_targets})
            selector = RandomSelector(torch.randn(1000, 16), cfg)
            source = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64)
            source_is_male = torch.tensor([1, 1, 0, 0, 0, 1], dtype=torch.bool)
            spec = torch.randn(6, 80, 100)
            target_1 = selector.select(spec, source, source_is_male)
            target_2 = selector.select(spec, source, source_is_male)

            all_same = torch.all(target_1 == target_2)
            if consistent_targets:
                self.assertTrue(all_same)
            else:
                self.assertFalse(all_same)

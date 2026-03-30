"""
Test whether the targets are consistent across the utterances of each speaker.
The base target selector is reponsible for this.
"""

import unittest
import json

import torch
from omegaconf import OmegaConf

from spkanon_eval.target_selection import RandomSelector


TARGET_DF = "spane/tests/datafiles/ls-dev-clean-2.txt"


class TestTargetSelection(unittest.TestCase):
    def test_consistent_selection(self):
        """
        Targets are consistent when each speaker is always assigned the same target.
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
        """Check that the same ID is avoided when the dataset is the same."""
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

    def test_target_constraint(self):
        """Filter targets by their metadata."""
        target_constraints = [
            {"is_male": False},
            {"fav_color": 1},
            {"fav_color": 1, "is_male": False},
        ]

        for constraint in target_constraints:
            cfg = OmegaConf.create(
                {
                    "consistent_targets": False,
                    "same_source_target": False,
                    "target_constraints": constraint,
                }
            )
            selector = RandomSelector(cfg, TARGET_DF)

            batch = {
                "feats": torch.randn(6, 10),
                "source": torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64),
            }

            # get filtered target speakers
            filtered_targets = set()
            for line in open(TARGET_DF):
                obj = json.loads(line)
                passes_constraints = True
                for key, value in constraint.items():
                    if obj[key] != value:
                        passes_constraints = False
                        break

                if passes_constraints:
                    filtered_targets.add(obj["speaker_id"])

            # check that only the filtered targets are available
            constraint_keys = list(constraint.keys())
            self.assertTrue(
                selector.target_info["speaker_id"].shape[0] == len(filtered_targets),
                constraint_keys,
            )
            for target in selector.target_info["speaker_id"]:
                self.assertTrue(target.item() in filtered_targets, constraint_keys)

            # check that only the filtered targets are returned
            for _ in range(3):
                targets = selector.select(batch)
                for selection in targets:
                    self.assertTrue(
                        selection.item() in filtered_targets, constraint_keys
                    )

    def test_conversion_constraint(self):
        """
        For each source, filter targets based on their respective values for some
        metadata.
        """
        conversion_constraints = [
            {"is_male": "same"},
            {"is_male": None},
            {"is_male": "opposite"},
            {"is_male": "same", "fav_color": "same"},
        ]

        for constraint in conversion_constraints:
            cfg = OmegaConf.create(
                {
                    "consistent_targets": False,
                    "same_source_target": False,
                    "conversion_constraints": constraint,
                }
            )
            selector = RandomSelector(cfg, TARGET_DF)

            batch = {
                "feats": torch.randn(6, 10),
                "source": torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.int64),
                "is_male": torch.tensor([True, True, False, False, False, True]),
                "fav_color": torch.tensor([1] * 6),
            }

            # get target speaker values for the relevant metadata
            target_metadata = {
                key: [None] * 3
                for key, value in constraint.items()
                if value is not None
            }
            for line in open(TARGET_DF):
                obj = json.loads(line)
                for key, value in constraint.items():
                    if value is None:
                        continue

                    target_metadata[key][obj["speaker_id"]] = obj[key]

            # check that only the valid targets are returned
            for _ in range(3):
                targets = selector.select(batch)
                self.assertTrue(batch["source"].shape == targets.shape)
                self.assertTrue(batch["source"].dtype == targets.dtype)

                for source_spkid, target_spkid in enumerate(targets):
                    for key, value in constraint.items():
                        if value is None:
                            continue

                        assert_func = (
                            self.assertTrue if value == "same" else self.assertFalse
                        )
                        assert_func(
                            batch[key][source_spkid]
                            == target_metadata[key][target_spkid],
                            constraint,
                        )

import os
import unittest
import shutil

import numpy as np
from sklearn.metrics import roc_curve

from spkanon_eval.evaluation.asv.asv_utils import analyse_results


class TestAsvAnalysis(unittest.TestCase):
    def setUp(self):
        # create/empty experiment folder
        self.exp_folder = "spane/tests/logs/asv_analysis"
        if os.path.isdir(self.exp_folder):
            shutil.rmtree(self.exp_folder)
        os.makedirs(os.path.join(self.exp_folder, "data"))

        # add the original datafile to the experiment folder
        datafile = "spane/tests/datafiles/ls-dev-clean-2.txt"
        shutil.copy(datafile, os.path.join(self.exp_folder, "data", "eval.txt"))

        # create a mock score file for trial-enroll pairs
        self.scores = np.array(
            [
                [0.0, 0.0, 0.8],
                [0.0, 1.0, 0.2],
                [0.0, 2.0, 0.4],
                [0.0, 0.0, 0.8],
                [0.0, 1.0, 0.4],
                [0.0, 2.0, 0.4],
                [1.0, 0.0, 0.8],
                [1.0, 1.0, 0.8],
                [1.0, 2.0, 0.8],
                [1.0, 0.0, 0.2],
                [1.0, 1.0, 0.4],
                [1.0, 2.0, 0.2],
                [2.0, 0.0, 0.2],
                [2.0, 1.0, 0.2],
                [2.0, 2.0, 0.8],
                [2.0, 0.0, 0.2],
                [2.0, 1.0, 0.2],
                [2.0, 2.0, 0.8],
            ]
        )
        os.makedirs(os.path.join(self.exp_folder, "results"))
        score_file = os.path.join(self.exp_folder, "results", "scores.npy")
        np.save(score_file, self.scores)

        # run the analysis
        analyse_results(datafile, score_file)

    def tearDown(self):
        """Remove the created directory."""
        shutil.rmtree(self.exp_folder)

    def test_eer(self):
        """Check that the EER for the whole file is computed correctly."""

        # check that the file with the overall EER exists
        eer_file = os.path.join(self.exp_folder, "results", "eer.txt")
        self.assertTrue(os.path.exists(eer_file))

        # check that the file content is correct
        line = open(eer_file).readlines()[1]
        n_pairs, threshold, eer = [float(v) for v in line.split()]
        self.assertTrue(int(n_pairs) == self.scores.shape[0])

        eer_expected, t_expected = get_eer_and_t(self.scores)
        self.assertTrue(np.isclose(threshold, t_expected))
        self.assertTrue(np.isclose(eer, eer_expected))

    def test_eer_gender(self):
        """
        Check that the gender analysis is done correctly. For each gender, two analysis
        should be performed: one where the group is evaluated in isolation, and once
        where the group is used to filter the trials only.
        """

        # get the gender of trials and enrolls
        spk_is_male = np.array([False, False, True], dtype=bool)
        trials_gender = self.scores[:, 0].astype(int)
        trials_gender = spk_is_male[trials_gender]
        enrolls_gender = self.scores[:, 1].astype(int)
        enrolls_gender = spk_is_male[enrolls_gender]

        # check the inter-variability analysis
        eer_file = os.path.join(self.exp_folder, "results", "eer_gender_inter.txt")
        self.assertTrue(os.path.exists(eer_file))

        results = dict()
        for line in open(eer_file).readlines()[1:]:
            gender, n_pairs, t, eer = line.split()
            results[gender] = [int(n_pairs), float(t), float(eer)]

        for gender_str, gender_bool in zip(["F", "M"], [False, True]):
            indices = np.where(trials_gender == gender_bool)[0]
            eer, t = get_eer_and_t(self.scores[indices])

            self.assertTrue(results[gender_str][0], indices.shape[0])
            self.assertTrue(np.isclose(results[gender_str][1], t))
            self.assertTrue(np.isclose(results[gender_str][2], eer))

        # check the intra-variability analysis
        eer_file = os.path.join(self.exp_folder, "results", "eer_gender_intra.txt")
        self.assertTrue(os.path.exists(eer_file))

        f_lines = open(eer_file).readlines()
        self.assertTrue(len(f_lines) == 2)  # there is only 1 male; cannot be evaluated
        gender, n_pairs, t, eer = f_lines[1].strip().split()
        self.assertTrue(gender == "F")

        indices = np.where(
            np.logical_and(trials_gender == False, enrolls_gender == False)
        )[0]
        eer_expected, t_expected = get_eer_and_t(self.scores[indices])

        self.assertTrue(int(n_pairs), indices.shape[0])
        self.assertTrue(np.isclose(float(t), t_expected))
        self.assertTrue(np.isclose(float(eer), eer_expected))

    def test_spk_identifiability(self):
        """
        Each speaker gets an identifiability score, which is the speaker's avg.
        score for same-speaker pairs, minus its avg. score for different-speaker pairs.
        Its pairs are those where the speaker is the trial. A higher score means that
        the speaker is easier to discriminate from the rest.
        """
        # check that the file exists
        eer_file = os.path.join(self.exp_folder, "results", "spk_identifiability.txt")
        self.assertTrue(os.path.exists(eer_file))

        # check that the file content is correct
        for line in open(eer_file).readlines()[1:]:
            spk, same_spk_pairs, diff_spk_pairs, score = line.strip().split()
            spk = float(spk)

            spk_scores = self.scores[self.scores[:, 0] == spk]
            same_spk_mean = np.mean(spk_scores[spk_scores[:, 1] == spk][:, 2])
            diff_spk_mean = np.mean(spk_scores[spk_scores[:, 1] != spk][:, 2])
            score_expected = np.round(same_spk_mean - diff_spk_mean, 3)

            self.assertTrue(int(same_spk_pairs) == np.sum(spk_scores[:, 1] == spk))
            self.assertTrue(int(diff_spk_pairs) == np.sum(spk_scores[:, 1] != spk))
            self.assertTrue(np.isclose(float(score), score_expected))

    def test_spk_eer(self):
        """Check the speaker-specific EERs."""
        # check that the file exists
        eer_file = os.path.join(self.exp_folder, "results", "eer_speaker.txt")
        self.assertTrue(os.path.exists(eer_file))

        # check that the file content is correct
        for line in open(eer_file).readlines()[1:]:
            spk, n_pairs, t, eer = line.strip().split()

            spk_scores = self.scores[self.scores[:, 0] == float(spk)]
            eer_expected, t_expected = get_eer_and_t(spk_scores)

            self.assertTrue(int(n_pairs) == spk_scores.shape[0])
            self.assertTrue(np.isclose(float(t), t_expected))
            self.assertTrue(np.isclose(float(eer), eer_expected))


def get_eer_and_t(scores: np.ndarray) -> tuple[float, float]:
    """Compute the EER and its threshold for the given scores."""
    same_speaker = scores[:, 0] == scores[:, 1]
    fpr, tpr, thresholds = roc_curve(same_speaker, scores[:, 2])
    eer_key = np.nanargmin(np.absolute(((1 - tpr) - fpr)))
    eer = np.round((fpr[eer_key] + (1 - tpr[eer_key])) / 2, 3)
    t = np.round(thresholds[eer_key], 3)
    return eer, t

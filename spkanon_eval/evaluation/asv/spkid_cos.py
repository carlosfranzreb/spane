"""
Similar to `splid_plda.py`, but using cosine similarity instead of PLDA as a backend.
Although not as effective, this approach does not require training a PLDA model,
making it faster and simpler to implement.

### Evaluation phase

1. Split the evaluation data into trial and enrollment data.
2. Use anonymized utterances of trial data, and maybe anonymized enrollment data.
3. Compute the SpkId vectors of all trial and enrollment utterances.
4. Compute LLRs of all pairs of trial and enrollment utterances with the trained PLDA
    model.
5. Average the LLRS across speakers.
6. Compute the ROC curve and the EER.

### Two attack scenarios

1. Ignorant: the attacker does not have access to the anonymization model. The training
    data of the ASV system and the enrollment data are not anonymized.
2. Semi-informed: the attacker has access to the anonymization model. The training data
    of the ASV system is anonymized without consistent targets, and the enrollment data
    is anonymized with consistent targets.
"""

import os
import logging

import numpy as np
from omegaconf import DictConfig

from spkanon_eval.anonymizer import Anonymizer
from .asv_utils import analyse_results, compute_dists
from .asv_component import ASVComponent


LOGGER = logging.getLogger("progress")
CHUNK_SIZE = 5000  # defines how many LLRs are computed in parallel


class FastASV(ASVComponent):
    def __init__(
        self, config: DictConfig, device: str, model: Anonymizer = None
    ) -> None:
        """Initialize the ASV system."""
        super().__init__(config, device, model)
        self.component_name = "asv-cos"

    def train(self, exp_folder: str) -> None:
        """This method does not require training."""
        LOGGER.warning("Fast ASV system does not require training")

    def evaluate(
        self, vecs: dict, labels: dict, dump_folder: str, datafile: str
    ) -> None:
        """
        Evaluate the ASV system on the given directory. The results of the
        anonymization are stored in the given path `exp_folder`, in the folder
        `data`. The results of this evaluation are stored in `eval`.
        The first utterance of each speaker is considered the trial utterances and
        the rest are considered the enrollment utterances.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
            is_baseline: whether the baseline is being evaluated
        """

        # compute spkemb distances of all pairs of trial and enrollment utterances
        dists, pairs = compute_dists(vecs, CHUNK_SIZE)
        del vecs

        # map utt indices to speaker indices
        pairs[:, 0] = labels["trials"][pairs[:, 0]]
        pairs[:, 1] = labels["enrolls"][pairs[:, 1]]

        # avg. dists across speakers and dump them to the experiment folder
        LOGGER.info("Averaging dists across speakers")
        LOGGER.info(f"No. of speaker pairs: {pairs.shape[0]}")
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        dist_avgs = np.bincount(inverse, weights=dists) / np.bincount(inverse)

        dist_file_utt = os.path.join(dump_folder, "utt_dists.npy")
        np.save(dist_file_utt, np.hstack((pairs, dists.reshape(-1, 1))))

        dist_file = os.path.join(dump_folder, "dists.npy")
        np.save(dist_file, np.hstack((unique_pairs, dist_avgs.reshape(-1, 1))))

        # compute the EER for the data and its subsets w.r.t. speaker chars.
        analyse_results(datafile, dist_file)

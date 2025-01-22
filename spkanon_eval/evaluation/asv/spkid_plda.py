"""
Slightly simplified version of the ASV system from the VPC 2022.

### Training phase

This phase requires a dataset different from the one used for evaluation.

1. Remove utterances that are too short, and speakers with too few utterances.
2. Fine-tune SpeechBrain's SpkId net.
3. Extract SpkId vectors from fine-tuned net.
4. Center the SpkId vectors.
5. Decrease the dimensionality of the centered SpkId vectors with LDA.
6. Train the PLDA model with the vectors resulting from LDA.

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
    is anonymized without consistent targets as well, since VPC24.
"""

import os
import pickle
import logging
import json

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import plda
from omegaconf import DictConfig

from spkanon_eval.anonymizer import Anonymizer
from .asv_utils import analyse_results, compute_llrs
from .asv_component import ASVComponent


LOGGER = logging.getLogger("progress")
CHUNK_SIZE = 5000  # defines how many LLRs are computed in parallel


class ASV(ASVComponent):
    def __init__(
        self, config: DictConfig, device: str, model: Anonymizer = None
    ) -> None:
        """Initialize the ASV system."""
        super().__init__(config, device, model)
        self.component_name = "asv-plda"

        # init the LDA model with ckpt if possible, else declare it null
        lda_ckpt = config.get("lda_ckpt", None)
        self.lda_model = None
        if lda_ckpt is not None:
            LOGGER.info(f"Loading LDA ckpt `{lda_ckpt}`")
            self.lda_model = pickle.load(open(lda_ckpt, "rb"))

        # init the PLDA model with ckpt if possible, else declare it null
        plda_ckpt = config.get("plda_ckpt", None)
        self.plda_model = None
        if plda_ckpt is not None:
            LOGGER.info(f"Loading PLDA ckpt `{plda_ckpt}`")
            self.plda_model = pickle.load(open(plda_ckpt, "rb"))

    def train(self, exp_folder: str) -> None:
        """
        Train the PLDA model with the SpkId vectors and also the SpkId model.
        The anonymized samples are stored in the given path `exp_folder`.
        If the scenario is "semi-informed", the training data is anonymized without
        consistent targets.
        """

        datafile = os.path.join(exp_folder, "data", "train_eval.txt")

        # define and create the directory where models and training data are stored
        dump_dir = os.path.join(
            exp_folder, "eval", "asv-plda", self.config.scenario, "train"
        )
        os.makedirs(dump_dir, exist_ok=True)

        # If the scenario is "semi-informed", anonymize the training data
        if self.config.scenario == "semi-informed":
            LOGGER.info(f"Anonymizing training data: {datafile}")
            datafile = self.anonymize_data(exp_folder, "train_eval", False)

        n_speakers = count_speakers(datafile)
        LOGGER.info(f"Number of speakers in training file: {n_speakers}")

        # fine-tune SpkId model and store the ckpt if needed
        if self.config.spkid.train:
            self.spkid_model.train(
                os.path.join(dump_dir, "spkid"), datafile, n_speakers
            )

        # compute SpkId vectors of all utterances with spkid model and center them
        vecs, labels = self.compute_spkid_vecs(datafile)
        train_mean_spkemb = np.mean(vecs, axis=0)
        vecs -= train_mean_spkemb

        # create the directory where models are stored
        models_dir = os.path.join(dump_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # perform dimensionality reduction with LDA if possible and store the model
        reduced_dims = self.config.get("reduced_dims", None)
        if reduced_dims is not None:
            if reduced_dims <= min(vecs.shape[1], n_speakers - 1):
                LOGGER.info(f"Reducing dimensionality to {reduced_dims} with LDA")
                self.lda_model = LinearDiscriminantAnalysis(n_components=reduced_dims)
                vecs = self.lda_model.fit_transform(vecs, labels)
                pickle.dump(
                    self.lda_model,
                    open(os.path.join(models_dir, "lda.pkl"), "wb"),
                )
            else:
                LOGGER.warning("Dimensionality reduction with LDA is not possible.")
                if reduced_dims <= vecs.shape[1]:
                    LOGGER.warning(
                        f"The reduction is larger than the num. of speakers ({n_speakers})."
                    )
                else:
                    LOGGER.warning(
                        f"Dim. {vecs.shape[1]} is lower than the desired reduction."
                    )

        # train the PLDA model and store the model
        LOGGER.info("Training PLDA model")
        self.plda_model = plda.Classifier()
        self.plda_model.train_mean_spkemb = train_mean_spkemb
        self.plda_model.fit_model(vecs, labels)
        if self.plda_model.model.pca is not None:
            n_components = self.plda_model.model.pca.components_.shape[0]
            LOGGER.warning(f"PCA is used within PLDA with {n_components} components")
        pickle.dump(self.plda_model, open(os.path.join(models_dir, "plda.pkl"), "wb"))

    def evaluate(
        self, vecs: dict, labels: dict, dump_folder: str, datafile: str
    ) -> None:
        """
        Evaluate the ASV system on the given directory. Each pair of trial
        and enrollment spkembs is given a log-likelihood ratio (LLR) by the
        PLDA model. These LLRs are dumped, and the EER is computed in
        `analyse_results`.
        """
        for name in ["trials", "enrolls"]:
            vecs[name] -= self.plda_model.train_mean_spkemb
            if self.lda_model is not None:
                vecs[name] = self.lda_model.transform(vecs[name])
            vecs[name] = self.plda_model.model.transform(
                vecs[name], from_space="D", to_space="U_model"
            )

        # compute LLRs of all pairs of trial and enrollment utterances
        llrs, pairs = compute_llrs(self.plda_model, vecs, CHUNK_SIZE)
        del vecs

        # map utt indices to speaker indices
        pairs[:, 0] = labels["trials"][pairs[:, 0]]
        pairs[:, 1] = labels["enrolls"][pairs[:, 1]]

        # avg. LLRs across speakers and dump them to the experiment folder
        LOGGER.info("Averaging LLRs across speakers")
        LOGGER.info(f"No. of speaker pairs: {pairs.shape[0]}")
        unique_pairs, inverse = np.unique(pairs, axis=0, return_inverse=True)
        llr_avgs = np.bincount(inverse, weights=llrs) / np.bincount(inverse)

        llr_file = os.path.join(dump_folder, "llrs.npy")
        np.save(llr_file, np.hstack((unique_pairs, llr_avgs.reshape(-1, 1))))

        # compute the EER for the data and its subsets w.r.t. speaker chars.
        analyse_results(datafile, llr_file)


def count_speakers(datafile: str) -> int:
    """Count the number of speakers in the datafile."""
    speakers = list()
    with open(datafile) as f:
        for line in f:
            obj = json.loads(line.strip())
            if obj["speaker_id"] not in speakers:
                speakers.append(obj["speaker_id"])

    n_speakers = len(speakers)
    LOGGER.info(f"Number of speakers in {datafile}: {n_speakers}")
    return n_speakers

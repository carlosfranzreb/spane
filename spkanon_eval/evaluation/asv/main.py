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
import copy
import pickle
import logging

import numpy as np
import plda
from omegaconf import DictConfig
from tqdm import tqdm

from spkanon_eval.anonymizer import Anonymizer
from spkanon_eval.component_definitions import EvalComponent
from spkanon_eval.setup_module import setup
from spkanon_eval.evaluate import SAMPLE_RATE
from spkanon_eval.inference import infer
from spkanon_eval.datamodules import setup_dataloader
from .trials_enrolls import split_trials_enrolls
from .asv_utils import analyse_results, compute_llrs, compute_dists, count_speakers


LOGGER = logging.getLogger("progress")
CHUNK_SIZE = 5000  # defines how many LLRs are computed in parallel


class ASV(EvalComponent):
    def __init__(
        self, config: DictConfig, device: str, model: Anonymizer = None
    ) -> None:
        """Initialize the ASV system."""
        if config.scenario == "semi-informed" and model is None:
            raise ValueError("Semi-informed scenario requires an anonymization system")
        if config.backend not in ["cos", "plda"]:
            raise ValueError(f"Invalid backend: {config.backend}")

        self.config = config
        self.device = device
        self.model = model
        self.spkid_model = setup(config.spkid, device)
        self.component_name = f"asv-{config.backend}"

        # init the PLDA model if needed
        if config.backend == "plda":
            plda_ckpt = config.get("plda_ckpt", None)
            self.plda_model = None
            if plda_ckpt is not None:
                LOGGER.info(f"Loading PLDA ckpt `{plda_ckpt}`")
                self.plda_model = pickle.load(open(plda_ckpt, "rb"))

        # if training is skipped, load the mean emb. of the training data
        self.train_mean_spkemb = None
        if config.train is False:
            self.train_mean_spkemb = np.load(config.train_mean_spkemb)

    def train(self, exp_folder: str) -> None:
        """
        Train the PLDA model with the SpkId vectors and also the SpkId model.
        The anonymized samples are stored in the given path `exp_folder`.
        If the scenario is "semi-informed", the training data is anonymized without
        consistent targets.
        """

        # define and create the directory where models and training data are stored
        datafile = os.path.join(exp_folder, "data", "train_eval.txt")
        dump_dir = os.path.join(
            exp_folder, "eval", self.component_name, self.config.scenario, "train"
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
        self.train_mean_spkemb = np.mean(vecs, axis=0)
        vecs -= self.train_mean_spkemb
        np.save(os.path.join(dump_dir, "train_mean_spkemb.npy"), self.train_mean_spkemb)

        # if cos. is used as a backend, we are done
        if self.config.backend == "cos":
            return

        # otherwise, train the PLDA model and store it
        LOGGER.info("Training PLDA model")
        n_components = self.config.get("reduce_dims", None)
        self.plda_model = plda.Classifier()
        self.plda_model.fit_model(vecs, labels, n_principal_components=n_components)
        if n_components is None and self.plda_model.model.pca is not None:
            n_components = self.plda_model.model.pca.components_.shape[0]
            LOGGER.warning(f"PCA is used within PLDA with {n_components} components")
        pickle.dump(self.plda_model, open(os.path.join(dump_dir, "plda.pkl"), "wb"))

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
            vecs[name] -= self.train_mean_spkemb
            if self.config.backend == "plda":
                vecs[name] = self.plda_model.model.transform(
                    vecs[name], from_space="D", to_space="U_model"
                )

        # average the enrollment speaker embeddings across speakers
        unique_speakers = np.unique(labels["enrolls"])
        avg_vecs = np.zeros((len(unique_speakers), vecs["enrolls"].shape[1]))

        for idx, speaker in enumerate(unique_speakers):
            speaker_indices = np.where(labels["enrolls"] == speaker)[0]
            avg_vecs[idx] = np.mean(vecs["enrolls"][speaker_indices], axis=0)

        vecs["enrolls"] = avg_vecs
        labels["enrolls"] = unique_speakers

        # compute scores of all pairs of trial and enrollment utterances
        if self.config.backend == "plda":
            scores, pairs = compute_llrs(self.plda_model, vecs, CHUNK_SIZE)
        else:
            scores, pairs = compute_dists(vecs, CHUNK_SIZE)
        del vecs

        # map utt indices to speaker indices and dump scores
        pairs[:, 0] = labels["trials"][pairs[:, 0]]
        pairs[:, 1] = labels["enrolls"][pairs[:, 1]]
        score_file = os.path.join(dump_folder, "scores.npy")
        np.save(score_file, np.hstack((pairs, scores.reshape(-1, 1))))

        # compute the EER for the data and its subsets w.r.t. speaker chars.
        analyse_results(datafile, score_file)

    def eval_dir(self, exp_folder: str, datafile: str, is_baseline: bool) -> None:
        """
        Split the evaluation data into trial and enrollment utterances, anonymize
        the enrollment data if necessary and compute and return the spkembs of all
        utterances. This is the shared part of the evaluation of the ASV for all
        ASV systems (spkid-plda and spkid-cos).
        """
        dump_folder = os.path.join(
            exp_folder, "eval", self.component_name, self.config.scenario
        )
        fname = os.path.splitext(os.path.basename(datafile))[0]
        dump_subfolder = os.path.join(dump_folder, "results", fname)
        os.makedirs(dump_subfolder, exist_ok=True)

        # split the datafile into trial and enrollment datafiles
        root_dir = None if is_baseline else self.config.data.config.root_folder
        anon_folder = self.config.data.config.get("anon_folder", None)
        anonymized_enrolls = self.config.inference.consistent_targets is False
        f_trials, f_enrolls = split_trials_enrolls(
            exp_folder,
            anonymized_enrolls,
            root_dir,
            anon_folder,
            self.config.data.datasets.get("enrolls", None),
        )

        # if the f_trials or f_enrolls do not exist, skip the evaluation
        if not (os.path.exists(f_trials) and os.path.exists(f_enrolls)):
            LOGGER.warning("No trials to evaluate; stopping component evaluation")
            return

        # Anonymize enrollment data if necessary
        if self.config.scenario == "semi-informed" and not anonymized_enrolls:
            LOGGER.info("Anonymizing enrollment data of the ASV system")
            f_enrolls = self.anonymize_data(exp_folder, "eval_enrolls", False)

        # compute SpkId vectors of all utts
        vecs, labels = dict(), dict()
        for name, f in zip(["trials", "enrolls"], [f_trials, f_enrolls]):
            vecs[name], labels[name] = self.compute_spkid_vecs(f)

            if self.config.get("save_spkembs", False) is True:
                np.save(os.path.join(exp_folder, f"{name}_spkembs.npy"), vecs)
                np.save(os.path.join(exp_folder, f"{name}_labels.npy"), labels)

        # call the child class to perform the evaluation
        self.evaluate(vecs, labels, dump_subfolder, datafile)

    def anonymize_data(
        self, exp_folder: str, df_name: str, consistent_targets: bool
    ) -> str:
        """
        Anonymize the given datafile and return the path to the anonymized datafile.

        Args:
            exp_folder: path to the experiment folder
            df_name: name of the datafile, without the directory or the extension
                (e.g. "f_enrolls"). The corresponding datafile is assumed to be in
                `{exp_folder}/data`.
            consistent_targets: whether each speaker should always be anonymized with
            the same target.
        """
        self.model.set_consistent_targets(consistent_targets)
        anon_datafile = infer(exp_folder, df_name, self.model, self.config)
        return anon_datafile

    def compute_spkid_vecs(self, datafile: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the SpkId vectors of all speakers and return them along with their
        speaker labels.

        Args:
            datafile: path to the datafile
            spkid_model: SpkId model
            config: configuration of the evaluation component
            sample_rate: sample rate of the data

        Returns:
            SpkId vectors: (n_utterances, embedding_dim)
            speaker labels: (n_utterances,)
        """
        LOGGER.info(f"Computing SpkId vectors of {datafile}")

        labels = np.array([], dtype=int)  # utterance labels
        vecs = None  # spkid vecs of utterances

        spkid_config = copy.deepcopy(self.config.data.config)
        spkid_config.batch_size = self.config.spkid.batch_size
        spkid_config.sample_rate = SAMPLE_RATE

        dl = setup_dataloader(self.spkid_model, spkid_config, datafile)
        for batch in tqdm(dl):
            new_vecs = self.spkid_model.run(batch).detach().cpu().numpy()
            new_vecs = np.nan_to_num(new_vecs)
            vecs = np.vstack([vecs, new_vecs]) if vecs is not None else new_vecs
            labels = np.concatenate([labels, batch[1].detach().cpu().numpy()])

        return vecs, labels

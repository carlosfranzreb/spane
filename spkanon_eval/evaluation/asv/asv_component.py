"""
Parent class for ASV evaluation, which includes the functionality that is common for
all ASV systems (spkid-plda and spkid-cos).
"""

import copy
import os
import logging

import numpy as np
from tqdm import tqdm

from spkanon_eval.evaluate import SAMPLE_RATE
from spkanon_eval.inference import infer
from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules import setup_dataloader
from spkanon_eval.component_definitions import EvalComponent
from .trials_enrolls import split_trials_enrolls


LOGGER = logging.getLogger("progress")


class ASVComponent(EvalComponent):
    def __init__(self, config, device, model=None):
        """Initialize the ASV system."""
        if config.scenario == "semi-informed" and model is None:
            raise ValueError("Semi-informed scenario requires a model")

        self.config = config
        self.device = device
        self.model = model
        self.spkid_model = setup(config.spkid, device)

    def train(self, exp_folder: str) -> None:
        """This method should be implemented by the child classes, if needed."""
        raise NotImplementedError()

    def evaluate(
        self, vecs: dict, labels: dict, dump_folder: str, datafile: str
    ) -> None:
        """
        This method is called by `eval_dir` to perform the ASV evaluation given the
        spkembs computed beforehand. It is expected to be implemented by the child
        classes.

        Args:
            vecs: the spkembs of the trial and enrollment utterances
            labels: the labels of the trial and enrollment utterances
            dump_folder: the folder where the results of the evaluation are stored
            datafile: the datafile that is being evaluated
        """
        raise NotImplementedError()

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

"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""

import pickle
import os
import copy
from shutil import rmtree
import json

import torchaudio
from omegaconf import OmegaConf
import numpy as np
import plda

from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules import setup_dataloader
from spkanon_eval.evaluate import SAMPLE_RATE as EVAL_SR
from spkanon_eval.utils import seed_everything

from base import BaseTestClass, run_pipeline


class TestEvalASV(BaseTestClass):
    def setUp(self):
        super().setUp()
        self.informed_config = OmegaConf.load("spane/config/components/asv/config.yaml")
        self.informed_config.asv.backend = "plda"
        self.informed_config.asv.train_spkid = False

        self.ignorant_config = self.informed_config.copy()
        self.ignorant_config.asv.scenario = "ignorant"
        self.spkemb_size = 192

    def tearDown(self):
        rmtree(self.init_config.log_dir)

    def test_results(self):
        """
        Test whether the ignorant ASV component, when given the ls-dev-clean-2 debug
        dataset for evaluation, yields the correct files, each with appropriate content,
        regardless of the specific EER values.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = self.ignorant_config
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "asv_test")
        config = run_pipeline(self.init_config)

        # assert that 3 files were created
        results_subdir = "eval/asv-plda/ignorant/results"
        results_dir = os.path.join(config.exp_folder, results_subdir)
        results_files = [f for f in os.listdir(results_dir) if f.endswith(".txt")]
        self.assertEqual(len(results_files), 3)

        # check the overall results
        with open(os.path.join(results_dir, "eer.txt")) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            out = lines[1].strip().split()
            self.assertEqual(out[0], "anon_eval")
            self.assertEqual(int(out[1]), 12)
            self.assertTrue(isinstance(float(out[2]), float))
            self.assertTrue(0 <= float(out[3]) <= 1)

    def test_pca_reduction(self):
        """
        Ensure that, when defined in the config, the PCA algorithm embedded in the PLDA
        package is trained and saved to disk, and that it is used to reduce the
        dimensionality of the speaker embeddings to the right dimension.

        ! n_components cannot be larger than min(n_features, n_classes - 1)
        """

        pca_output_size = 2
        self.init_config.eval.components = self.ignorant_config.copy()
        self.init_config.eval.components.asv.reduce_dims = pca_output_size
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_pca_reduction"
        )
        config = run_pipeline(self.init_config)

        asv_subdir = "eval/asv-plda/ignorant"
        asv_dir = os.path.join(config.exp_folder, asv_subdir)

        # assert that the input and output sizes of the PCA model are correct
        plda_path = os.path.join(asv_dir, "train", "plda.pkl")
        plda = pickle.load(open(plda_path, "rb"))

        x = np.random.randn(2, self.spkemb_size)
        pca_out = plda.model.pca.transform(x)
        self.assertTrue(pca_out.shape == (2, 2), "PCA output shape is incorrect")

    def test_enrollment_targets(self):
        """
        Ensure that, when the inference and evaluation seeds are different, the targets
        chosen for inference and enrollment utterances of each source speaker also
        differ.

        This is important to ensure that enrollment speakers are not anonymized with
        the same targets that were already used during inference, which would make the
        ASV evaluation trivial, as the ASV system would detect targets instead of
        sources.

        This is only necessary when `consistent_targets` is true, i.e. speaker-level
        target selection. When `consistent_targets` is false, we don't need to
        anonymize the enrollment speakers again.
        """

        # add the dummy featproc component to the config and random target selection
        self.init_config.featproc.dummy.n_targets = 20
        self.init_config.eval.config.seed = self.init_config.seed + 100
        self.init_config.eval.components = self.informed_config
        self.init_config.eval.components.asv.consistent_targets = True
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_enrollment_targets"
        )

        # run the experiment and get the log file with the selected targets
        config = run_pipeline(self.init_config)

        # gather the source-target pairs, separating them by run (inference, enroll)
        targets = list()
        for f in ["anon_eval.txt", "anon_eval_enrolls.txt"]:
            targets.append(list())
            for line in open(os.path.join(config.exp_folder, "data", f)):
                obj = json.loads(line)
                targets[-1].append((obj["speaker_id"], obj["target"]))

        # assert that there are two lists of targets: inference and enrollment
        self.assertEqual(len(targets), 2)

        # assert that the targets are different
        found_difference = False
        for infer_pair in targets[0]:
            for enroll_pair in targets[1]:
                if infer_pair[0] == enroll_pair[0]:
                    if infer_pair[1] != enroll_pair[1]:
                        found_difference = True
                        break
            if found_difference is True:
                break
        self.assertTrue(
            found_difference, "All inference and enrollment targets are the same"
        )

    def test_informed_asv(self):
        """
        In the semi-informed scenario, the ASV system is trained with anonymized
        enrollment utterances. Assert that they are anonymized by ensuring that:
        1. the anonymized utterances differ from the original ones,
        2. that the ASV sytem was trained with the anonymized utterances

        We assume that LibriSpeech's dev-clean-2 dataset is used for training.
        """

        # add the dummy featproc component to the config and random target selection
        self.init_config.target_selection = {
            "cls": "spkanon_eval.target_selection.random.RandomSelector",
            "consistent_targets": True,
        }
        self.init_config.eval.config.seed = self.init_config.seed + 1
        self.init_config.eval.components = self.informed_config
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_informed"
        )
        config = run_pipeline(self.init_config)
        asv_dir = os.path.join(config.exp_folder, "eval", "asv-plda", "semi-informed")

        # find the train_eval datafile and assert that it is LibriSpeech's dev-clean-2
        train_files = config.data.datasets.train_eval
        self.assertEqual(len(train_files), 1, "Wrong number of train_eval datasets")

        train_file = train_files[0]
        self.assertEqual(
            train_file,
            "spane/data/debug/ls-dev-clean-2.txt",
            "Wrong train_eval dataset",
        )

        anon_train_file = os.path.join(config.exp_folder, "data", "anon_train_eval.txt")
        self.assertTrue(
            os.path.exists(anon_train_file),
            "The anonymized train_eval file does not exist",
        )

        anon_root = os.path.join(asv_dir, "train")
        orig_root = os.path.join("tests", "data")
        data_dir = os.path.join("LibriSpeech", "dev-clean-2")

        # assert that the anonymized utterances differ from the original ones
        for root, _, file in os.walk(os.path.join(anon_root, data_dir)):
            for f in file:
                anon_path = os.path.join(root, f)
                orig_path = os.path.join(root.replace(anon_root, orig_root), f)
                anon_utt, anon_sr = torchaudio.load(anon_path)
                orig_utt, orig_sr = torchaudio.load(orig_path)
                self.assertNotEqual(anon_sr, orig_sr, "The sample rates are the same")
                self.assertNotEqual(
                    list(anon_utt.shape),
                    list(orig_utt.shape),
                    "The shapes are the same",
                )

        # set the same seed as in the experiment
        seed_everything(self.init_config.eval.config.seed)

        # compute the spkid vecs for the anonymized utterances
        # as is done in spkanon_eval.evaluation.asv.spkid_plda.compute_spkid_vecs
        spkid_model = setup(config.eval.components.asv.spkid, "cpu")
        labels = np.array([], dtype=int)  # utterance labels
        vecs = None  # spkid vecs of utterances

        spkid_config = copy.deepcopy(config.data.config)
        spkid_config.sample_rate = EVAL_SR
        dl = setup_dataloader(spkid_model, spkid_config, anon_train_file)
        for batch in dl:
            new_vecs = spkid_model.run(batch).detach().cpu().numpy()
            vecs = new_vecs if vecs is None else np.vstack([vecs, new_vecs])
            new_labels = batch[1].detach().cpu().numpy()
            labels = np.concatenate([labels, new_labels])
        vecs -= np.mean(vecs, axis=0)

        # assert that the number of vectors matches the number of lines in the train file
        n_lines = len(open(anon_train_file).readlines())
        self.assertEqual(vecs.shape[0], n_lines, "Wrong number of spkid vecs")

        # train a PLDA algorithm with the spkid vecs and compare it to the one used in
        # the ASV system
        new_plda = plda.Classifier()
        new_plda.fit_model(vecs, labels)
        old_plda = pickle.load(open(os.path.join(asv_dir, "train", "plda.pkl"), "rb"))

        for attr in ["m", "A", "Psi", "relevant_U_dims", "inv_A"]:
            self.assertTrue(
                np.allclose(
                    getattr(new_plda.model, attr),
                    getattr(old_plda.model, attr),
                ),
                f"The attribute {attr} of the PLDA models differ",
            )

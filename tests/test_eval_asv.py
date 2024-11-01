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

from base import BaseTestClass, run_pipeline
from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules import setup_dataloader
from spkanon_eval.evaluate import SAMPLE_RATE as EVAL_SR
from spkanon_eval.utils import seed_everything


SPKEMB_SIZE = 192
SPKID_CONFIG = {
    "cls": "spkanon_eval.featex.spkid.spkid.SpkId",
    "path": "speechbrain/spkrec-ecapa-voxceleb",
    "train": False,
    "batch_size": 2,
    "num_workers": 0,
    "emb_model_ckpt": None,
}
ASV_IGNORANT_CONFIG = OmegaConf.create(
    {
        "asv_ignorant": {
            "cls": "spkanon_eval.evaluation.asv.spkid_plda.ASV",
            "scenario": "ignorant",
            "train": True,
            "lda_ckpt": None,
            "plda_ckpt": None,
            "spkid": SPKID_CONFIG,
        }
    }
)
ASV_LAZY_CONFIG = OmegaConf.create(
    {
        "asv_lazy_informed": {
            "cls": "spkanon_eval.evaluation.asv.spkid_plda.ASV",
            "scenario": "lazy-informed",
            "train": True,
            "inference": "${inference}",
            "sample_rate": "${synthesis.sample_rate}",
            "lda_ckpt": None,
            "plda_ckpt": None,
            "spkid": SPKID_CONFIG,
        }
    }
)
FEATPROC_CONFIG = {
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
    "output": {"featproc": ["spectrogram", "n_frames", "target"], "featex": []},
}


class TestEvalASV(BaseTestClass):
    def test_results(self):
        """
        Test whether the ignorant ASV component, when given the ls-dev-clean-2 debug
        dataset for evaluation, yields the correct files, each with appropriate content,
        regardless of the specific EER values.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = ASV_IGNORANT_CONFIG
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "asv_test")
        config, log_dir = run_pipeline(self.init_config)

        # assert that 3 files were created
        results_subdir = "eval/asv-plda/ignorant/results"
        results_dir = os.path.join(log_dir, results_subdir)
        results_files = [f for f in os.listdir(results_dir) if f.endswith(".txt")]
        self.assertEqual(len(results_files), 3)

        # check the overall results
        with open(os.path.join(results_dir, "eer.txt")) as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 2)
            out = lines[1].strip().split()
            self.assertEqual(out[0], "anon_eval")
            self.assertEqual(int(out[1]), 9)
            self.assertTrue(isinstance(float(out[2]), float))
            self.assertTrue(0 <= float(out[3]) <= 1)

        rmtree(self.init_config.log_dir)

    def test_lda_reduction(self):
        """
        Ensure that, when defined in the config, an LDA algorithm is trained and saved
        to disk, and that it is used to reduce the dimensionality of the speaker
        embeddings to the right dimension.

        ! n_components cannot be larger than min(n_features, n_classes - 1)
        """

        lda_output_size = 2
        self.init_config.eval.components = ASV_IGNORANT_CONFIG
        self.init_config.eval.components.asv_ignorant.reduced_dims = lda_output_size
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_lda_reduction"
        )
        config, log_dir = run_pipeline(self.init_config)

        asv_subdir = "eval/asv-plda/ignorant"
        asv_dir = os.path.join(log_dir, asv_subdir)

        # assert that the LDA algorithm was trained and saved to disk
        lda_path = os.path.join(asv_dir, "train", "models", "lda.pkl")
        self.assertTrue(
            os.path.exists(os.path.join(asv_dir, "train", "models", "lda.pkl"))
        )

        # assert that the input and output sizes of the LDA model are correct
        lda = pickle.load(open(lda_path, "rb"))
        x = np.random.randn(2, SPKEMB_SIZE)
        try:
            lda_out = lda.transform(x)
        except ValueError:
            self.fail("The input size of the LDA model is incorrect")

        self.assertEqual(
            lda_out.shape[1],
            lda_output_size,
            "The output size of the LDA model is incorrect",
        )

        # assert that the PLDA model does not use PCA internally and that it expects
        # the LDA input size
        plda_path = os.path.join(asv_dir, "train", "models", "plda.pkl")
        plda = pickle.load(open(plda_path, "rb"))
        self.assertEqual(plda.model.pca, None, "The PLDA model uses PCA internally")
        self.assertEqual(
            plda.model.get_dimensionality("U"),
            lda_output_size,
            "The PLDA model expects a different input size",
        )

        rmtree(self.init_config.log_dir)

    def test_enrollment_targets(self):
        """
        Ensure that, when the inference and evaluation seeds are different, the targets
        chosen for inference and enrollment utterances of each source speaker also
        differ. We test this only with the Common Voice source speakers.

        This is important to ensure that enrollment speakers are not anonymized with
        the same targets that were already used during inference, which would make the
        ASV evaluation trivial, as the ASV system would detect target instead of source
        speakers.
        """

        # add the dummy featproc component to the config and random target selection
        self.init_config.target_selection = {
            "cls": "spkanon_eval.target_selection.random.RandomSelector",
            "consistent_targets": True,
        }
        self.init_config.featproc = FEATPROC_CONFIG
        self.init_config.eval.config.seed = self.init_config.seed + 100
        self.init_config.eval.components = ASV_LAZY_CONFIG
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_enrollment_targets"
        )

        # run the experiment and get the log file with the selected targets
        config, log_dir = run_pipeline(self.init_config)

        # gather the source-target pairs, separating them by run (inference, enroll)
        targets = list()
        for f in ["anon_eval.txt", "anon_eval_enrolls.txt"]:
            targets.append(list())
            for line in open(os.path.join(log_dir, "data", f)):
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

        rmtree(self.init_config.log_dir)

    def test_lazy_informed_asv(self):
        """
        In the lazy-informed scenario, the ASV system is trained with anonymized
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
        self.init_config.featproc = FEATPROC_CONFIG
        self.init_config.eval.config.seed = self.init_config.seed + 1
        self.init_config.eval.components = ASV_LAZY_CONFIG
        self.init_config.log_dir = os.path.join(
            self.init_config.log_dir, "asv_test_lazy_informed"
        )
        config, log_dir = run_pipeline(self.init_config)
        asv_dir = os.path.join(log_dir, "eval", "asv-plda", "lazy-informed")

        # find the train_eval datafile and assert that it is LibriSpeech's dev-clean-2
        train_files = config.data.datasets.train_eval
        self.assertEqual(len(train_files), 1, "Wrong number of train_eval datasets")

        train_file = train_files[0]
        self.assertEqual(
            train_file,
            "spkanon_eval/data/debug/ls-dev-clean-2.txt",
            "Wrong train_eval dataset",
        )

        anon_train_file = os.path.join(log_dir, "data", "anon_train_eval.txt")
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
        spkid_model = setup(config.eval.components.asv_lazy_informed.spkid, "cpu")
        labels = np.array([], dtype=int)  # utterance labels
        vecs = None  # spkid vecs of utterances

        spkid_config = copy.deepcopy(config.data.config)
        spkid_config.batch_size = SPKID_CONFIG["batch_size"]
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
        old_plda = pickle.load(
            open(os.path.join(asv_dir, "train", "models", "plda.pkl"), "rb")
        )

        for attr in ["m", "A", "Psi", "relevant_U_dims", "inv_A"]:
            self.assertTrue(
                np.allclose(
                    getattr(new_plda.model, attr),
                    getattr(old_plda.model, attr),
                ),
                f"The attribute {attr} of the PLDA models differ",
            )

        rmtree(self.init_config.log_dir)

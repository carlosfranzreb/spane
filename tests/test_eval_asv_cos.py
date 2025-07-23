"""
Test the evaluation components. We don't check whether the numbers are right, like the
EER or the LLRs, but rather that these numbers are computed for the correct speakers
and utterances. This test class inherits from BaseTestClass, which runs the inference
for the debug data.
"""

import os
from shutil import rmtree


from omegaconf import OmegaConf
import numpy as np

from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules import setup_dataloader
from spkanon_eval.utils import seed_everything
from spkanon_eval.evaluate import SAMPLE_RATE as EVAL_SR

from base import BaseTestClass, run_pipeline


SPKEMB_SIZE = 192
SPKID_CONFIG = {
    "cls": "spkanon_eval.featex.SpkId",
    "path": "speechbrain/spkrec-xvect-voxceleb",
    "ckpt": None,
    "train": False,
    "num_workers": 0,
    "train_config": "spkanon_eval/config/components/asv/spkid/train_xvector.yaml",
}
ASV_IGNORANT_CONFIG = OmegaConf.create(
    {
        "asv_ignorant": {
            "cls": "spkanon_eval.evaluation.ASV",
            "scenario": "ignorant",
            "spkid": SPKID_CONFIG,
            "train": False,
            "backend": "cos",
            "train_mean_spkemb": None,
            "plda_ckpt": None,
            "save_spkembs": False,
            "consistent_targets": False,
            "sample_rate_out": 16000,
        }
    }
)


class TestEvalASVCos(BaseTestClass):
    def test_results(self):
        """
        Test whether the ignorant ASV component, when given the ls-dev-clean-2 debug
        dataset for evaluation, yields the correct files, each with appropriate content,
        regardless of the specific EER values.
        """

        # run the experiment with the ignorant attack scenario
        self.init_config.eval.components = ASV_IGNORANT_CONFIG
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "asv_test")
        config = run_pipeline(self.init_config)

        # assert that 3 files were created
        results_subdir = "eval/asv-cos/ignorant/results"
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

        rmtree(self.init_config.log_dir)

    def test_cos_similarities(self):
        """
        Check that the cosine similarities are computed correctly by computing them
        manually and comparing them to the ones computed by the ASV component.
        """

        # run the experiment with the ignorant attack scenario
        self.init_config.eval.components = ASV_IGNORANT_CONFIG
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "asv_test")
        config = run_pipeline(self.init_config)

        # load the computed cosine similarities
        results_subdir = "eval/asv-cos/ignorant/results/anon_eval"
        scores_file = os.path.join(config.exp_folder, results_subdir, "scores.npy")
        self.assertTrue(os.path.exists(scores_file))
        scores = np.load(scores_file)

        # set the same seed as in the experiment
        seed_everything(self.init_config.seed)

        # compute the spkembs of the trial and enrollment utterances
        spkid_model = setup(OmegaConf.create(SPKID_CONFIG), "cpu")
        df_cfg = OmegaConf.create(
            {
                "sample_rate_in": EVAL_SR,
                "num_workers": 0,
            }
        )
        vecs = {"trials": None, "enrolls": None}
        speakers = {"trials": None, "enrolls": None}
        for key in vecs:
            df = os.path.join(config.exp_folder, "data", f"eval_{key}.txt")
            dl = setup_dataloader(spkid_model, df_cfg, df)
            for batch in dl:
                batch_vecs = spkid_model.run(batch).detach().cpu().numpy()
                vecs[key] = (
                    np.concatenate((vecs[key], batch_vecs), axis=0)
                    if vecs[key] is not None
                    else batch_vecs
                )
                speakers[key] = (
                    np.concatenate((speakers[key], batch[1]), axis=0)
                    if speakers[key] is not None
                    else batch[1]
                )
            self.assertEqual(len(vecs[key]), len(open(df).readlines()))

        # average the enrollment speaker embeddings across speakers
        unique_speakers = np.unique(speakers["enrolls"])
        avg_vecs = np.zeros((len(unique_speakers), vecs["enrolls"].shape[1]))

        for idx, speaker in enumerate(unique_speakers):
            speaker_indices = np.where(speakers["enrolls"] == speaker)[0]
            avg_vecs[idx] = np.mean(vecs["enrolls"][speaker_indices], axis=0)

        vecs["enrolls"] = avg_vecs
        speakers["enrolls"] = unique_speakers

        # compute the cosine similarities manually
        write_idx = 0
        test_scores = np.zeros((vecs["trials"].shape[0] * vecs["enrolls"].shape[0], 3))
        for enroll_idx, enroll_vec in enumerate(vecs["enrolls"]):
            for trial_idx, trial_vec in enumerate(vecs["trials"]):
                test_scores[write_idx, 0] = speakers["trials"][trial_idx]
                test_scores[write_idx, 1] = speakers["enrolls"][enroll_idx]
                test_scores[write_idx, 2] = np.dot(trial_vec, enroll_vec) / (
                    np.linalg.norm(trial_vec) * np.linalg.norm(enroll_vec)
                )
                write_idx += 1

        # compare the computed cosine similarities
        self.assertTrue(np.allclose(test_scores, scores))

        rmtree(self.init_config.log_dir)

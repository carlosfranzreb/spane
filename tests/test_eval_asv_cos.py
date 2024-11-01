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

from base import BaseTestClass, run_pipeline
from spkanon_eval.setup_module import setup
from spkanon_eval.datamodules import setup_dataloader
from spkanon_eval.utils import seed_everything
from spkanon_eval.evaluate import SAMPLE_RATE as EVAL_SR


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
            "cls": "spkanon_eval.evaluation.asv.spkid_cos.FastASV",
            "scenario": "ignorant",
            "spkid": SPKID_CONFIG,
            "train": False,
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
        config, log_dir = run_pipeline(self.init_config)

        # assert that 3 files were created
        results_subdir = "eval/asv-cos/ignorant/results"
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

    def test_cos_similarities(self):
        """
        Check that the cosine similarities are computed correctly by computing them
        manually and comparing them to the ones computed by the ASV component.
        """

        # run the experiment with the ignorant attack scenario
        self.init_config.eval.components = ASV_IGNORANT_CONFIG
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "asv_test")
        config, log_dir = run_pipeline(self.init_config)

        # load the computed cosine similarities
        results_subdir = "eval/asv-cos/ignorant/results/anon_eval"
        dists_file = os.path.join(log_dir, results_subdir, "dists.npy")
        self.assertTrue(os.path.exists(dists_file))
        dists = np.load(dists_file)

        # set the same seed as in the experiment
        seed_everything(self.init_config.seed)

        # compute the spkembs of the trial and enrollment utterances
        spkid_model = setup(OmegaConf.create(SPKID_CONFIG), "cpu")
        df_cfg = OmegaConf.create(
            {
                "sample_rate": EVAL_SR,
                "batch_size": 2,
                "num_workers": 0,
                "chunk_sizes": {"eval": {100: 1}},
            }
        )
        vecs = {"trials": None, "enrolls": None}
        speakers = {"trials": None, "enrolls": None}
        for key in vecs:
            df = os.path.join(log_dir, "data", f"eval_{key}.txt")
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

        # compute the cosine similarities manually
        test_dists_utt = np.zeros(
            (vecs["trials"].shape[0] * vecs["enrolls"].shape[0], 3)
        )
        for trial_idx, trial_vec in enumerate(vecs["trials"]):
            for enroll_idx, enroll_vec in enumerate(vecs["enrolls"]):
                write_idx = trial_idx * vecs["enrolls"].shape[0] + enroll_idx
                test_dists_utt[write_idx, 0] = speakers["trials"][trial_idx]
                test_dists_utt[write_idx, 1] = speakers["enrolls"][enroll_idx]
                test_dists_utt[write_idx, 2] = np.dot(trial_vec, enroll_vec) / (
                    np.linalg.norm(trial_vec) * np.linalg.norm(enroll_vec)
                )

        # average the cosine similarities across speakers
        n_speakers = speakers["trials"].shape[0]
        test_dists = np.zeros((n_speakers * n_speakers, 3))
        for trial_spk in range(n_speakers):
            for enroll_spk in range(n_speakers):
                write_idx = trial_spk * n_speakers + enroll_spk
                utt_indices = np.where(
                    (test_dists_utt[:, 0] == trial_spk)
                    & (test_dists_utt[:, 1] == enroll_spk)
                )[0]
                test_dists[write_idx, 0] = trial_spk
                test_dists[write_idx, 1] = enroll_spk
                test_dists[write_idx, 2] = np.mean(test_dists_utt[utt_indices, 2])

        # compare the computed cosine similarities
        self.assertTrue(np.allclose(test_dists, dists))

        rmtree(self.init_config.log_dir)

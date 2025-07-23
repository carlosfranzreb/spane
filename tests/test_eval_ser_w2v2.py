import os
import json
import shutil

import torch
from omegaconf import OmegaConf

from spkanon_eval.evaluation import EmotionEvaluator
from base import BaseTestClass, run_pipeline


class TestEvalSer(BaseTestClass):
    def test_results(self):
        """
        Test whether the SER results are valid. They should be between 0 and 1.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = {
            "audeering_w2v": {
                "cls": "spkanon_eval.evaluation.ser.audeering_w2v.EmotionEvaluator",
                "init": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "train": False,
                "num_workers": 0,
                "root_folder": "${data.config.root_folder}",
            }
        }
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "eval_ser")
        config = run_pipeline(self.init_config)
        results_dir = os.path.join(config.exp_folder, "eval", "ser-audeering-w2v")
        self.assertTrue(os.path.isdir(results_dir))

        # gather the utterances from the datafile
        expected_utts = list()
        for line in open(os.path.join(config.data.datasets.eval[0])):
            expected_utts.append(json.loads(line)["path"].replace("{root}/", ""))

        # check the results
        with open(os.path.join(results_dir, "anon_eval.txt")) as f:
            results = f.readlines()
            self.assertEqual(len(results), len(expected_utts) + 1)
            for line in results[1:]:
                values = line.split()
                self.assertTrue(len(values), 8)
                fname = values[0][values[0].index("LibriSpeech") :]
                self.assertTrue(fname in expected_utts)
                for idx in range(1, 8):
                    self.assertTrue(-1.1 <= float(values[idx]) <= 1.1)

        shutil.rmtree(self.init_config.log_dir)

    def test_batch(self):
        """
        Test whether the batching works: the results of the batch should equal the results of the
        individual samples. We check the emotion embeddings for each sample.
        """
        audios = torch.randn(4, 50000) - 0.5
        config = OmegaConf.create(
            {
                "init": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "batch_size": 2,
                "data": {
                    "config": {
                        "trainer": {"batch_size": 2},
                        "sample_rate": 16000,
                    },
                },
            }
        )
        evaluator = EmotionEvaluator(config, "cpu")
        batched_out = evaluator.run([audios])[0]
        single_out = [evaluator.run([audio.unsqueeze(0)])[0] for audio in audios]
        for i in range(len(batched_out)):
            self.assertTrue(torch.allclose(batched_out[i], single_out[i]))
            if i > 0:
                self.assertFalse(torch.allclose(batched_out[i], batched_out[i - 1]))

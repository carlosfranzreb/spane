import os
import json
import shutil

import torch
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

from spkanon_eval.evaluation import EmotionEvaluator
from spkanon_eval.datamodules import AudioBatch

from base import BaseTestClass, run_pipeline


class TestEvalSer(BaseTestClass):
    def test_results(self):
        """
        Test whether the SER results are valid. They should be between 0 and 1.
        """

        # run the experiment with both ASV evaluation scenarios
        self.init_config.eval.components = OmegaConf.load(
            "spane/config/components/ser/audeering_w2v.yaml"
        )
        self.init_config.log_dir = os.path.join(self.init_config.log_dir, "eval_ser")
        config = run_pipeline(self.init_config)
        results_dir = os.path.join(config.exp_folder, "eval", "ser-audeering-w2v")
        self.assertTrue(os.path.isdir(results_dir))

        # gather the utterances from the datafile
        expected_utts = list()
        for line in open(os.path.join(config.data.datasets.eval[0])):
            expected_utts.append(
                json.loads(line)["path"].replace("spane/tests/data/", "")
            )

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
        Test whether the batching works: the results of the batch should equal the
        results of the individual samples. We check the emotion embeddings for each
        sample.
        """
        lens = torch.tensor([50000, 40000, 45000, 32000])
        spkid = torch.tensor([0, 0, 0, 0])
        audios = [torch.randn(len) - 0.2 for len in lens]
        audios_batch = pad_sequence(audios, batch_first=True)
        batch = AudioBatch(audios_batch, spkid, lens)

        # init the model and run the batch
        config = OmegaConf.create(
            {
                "init": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                "data": {"config": {"sample_rate": 16000}},
            }
        )
        evaluator = EmotionEvaluator(config, "cpu")
        batched_out = evaluator.run(batch)[0]

        # run the audios one by one
        single_out = list()
        for idx in range(len(audios)):
            single_batch = AudioBatch(
                audios[idx].unsqueeze(0),
                spkid[idx : idx + 1],
                lens[idx : idx + 1],
            )
            single_out.append(evaluator.run(single_batch)[0].squeeze(0))

        # compare the two outputs
        for i in range(len(single_out)):
            self.assertTrue(torch.allclose(batched_out[i], single_out[i], atol=1e-4))
            if i > 0:
                self.assertFalse(
                    torch.allclose(batched_out[i], batched_out[i - 1], atol=1e-4)
                )

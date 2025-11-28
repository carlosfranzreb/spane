"""
Self-SSL model for Parkinson's disease detection from

<https://github.com/david-gimeno/interpreting-ssl-parkinson-speech>
"""

import os
import logging

import torch
import numpy as np
from tqdm import tqdm

from spkanon_eval.evaluate import SAMPLE_RATE
from spkanon_eval.evaluation.pdd.analysis_utils import analyse_func, headers_func
from spkanon_eval.evaluation.analysis import analyse_results
from spkanon_eval.datamodules import eval_dataloader
from spkanon_eval.component_definitions import InferComponent, EvalComponent

from .model import PdDetector

LOGGER = logging.getLogger("progress")


class PdEvaluator(InferComponent, EvalComponent):
    def __init__(self, config, device, **kwargs):
        self.config = config
        self.config.data.config.sample_rate_out = SAMPLE_RATE
        self.device = device

        self.model = PdDetector(config.ckpt)
        self.model.eval()
        self.model.to(device)

    def to(self, device):
        self.device = device
        self.model.to(device)

    @torch.inference_mode()
    def run(self, batch):
        return self.model(batch[0].to(self.device), batch[2])

    def train(self, exp_folder, datafiles):
        raise NotImplementedError

    def eval_dir(self, exp_folder: str, datafile: str, is_baseline: bool) -> None:
        """
        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
            is_baseline: whether original data is being evaluated.
        """
        eval_dir = "pd_detection"
        if is_baseline:
            eval_dir += "-baseline"

        dump_folder = os.path.join(exp_folder, "eval", eval_dir)
        os.makedirs(dump_folder, exist_ok=True)

        # define the dump file and write the headers
        x, y = list(), list()
        dump_file = os.path.join(dump_folder, os.path.basename(datafile))
        with open(dump_file, "w", encoding="utf-8") as f:
            f.write("path n_edits n_words_ref wer text\n")

        for batch, sample_data in tqdm(
            eval_dataloader(self.config.data.config, datafile, self)
        ):
            batch_out = self.run(batch)
            batch_out = batch_out.argmax(dim=1)
            for idx, out in enumerate(batch_out):  # iterate through the batch
                audiofile = sample_data[idx]["path"]
                y.append(sample_data[idx]["pd"])
                x.append(out.item())

                # dump the results for this sample into the dump file
                with open(dump_file, "a", encoding="utf-8") as f:
                    f.write(f"{audiofile} {y[-1]} {out}\n")

        analyse_results(
            dump_folder, datafile, np.array([x, y]).T, analyse_func, headers_func
        )

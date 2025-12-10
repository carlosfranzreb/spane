"""
Self-SSL model for Parkinson's disease detection from

<https://github.com/david-gimeno/interpreting-ssl-parkinson-speech>

TODO:

1. I'm currently evaluating on all of the data, but 4/5ths of it are used for training.
"""

import os
import logging

import torch
from torch import Tensor
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

        self.model = PdDetector(config.ckpt_dir, device)
        self.model.eval()
        self.model.to(device)

    def to(self, device: str):
        self.device = device
        self.model.to(device)

    @torch.inference_mode()
    def run(self, batch: list[Tensor], folds: Tensor) -> Tensor:
        return self.model(batch[0], batch[2], folds)

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
            f.write("path label prediction\n")

        for batch, sample_data in tqdm(
            eval_dataloader(self.config.data.config, datafile, self)
        ):
            folds = torch.tensor([d["fold"] for d in sample_data])
            batch_out = self.run(batch, folds)
            batch_out = batch_out.argmax(dim=1)
            for idx, out in enumerate(batch_out):  # iterate through the batch
                audiofile = sample_data[idx]["path"]
                y.append(sample_data[idx]["pd"])
                x.append(out.item())

                # dump the results for this sample into the dump file
                with open(dump_file, "a", encoding="utf-8") as f:
                    f.write(f"{audiofile} {str(int(y[-1]))} {out}\n")

        analyse_results(
            dump_folder, datafile, np.array([x, y]).T, analyse_func, headers_func
        )

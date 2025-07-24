import os
import logging
import inspect
import json

import torch
import pandas as pd

from NISQA.nisqa.NISQA_lib import NISQA, SpeechQualityDataset, predict_mos
from spkanon_eval.evaluation.naturalness.analysis_utils import (
    analyse_func,
    headers_func,
)
from spkanon_eval.evaluation.analysis import analyse_results
from spkanon_eval.component_definitions import EvalComponent


LOGGER = logging.getLogger("progress")


class NisqaEvaluator(EvalComponent):
    def __init__(self, config, device, **kwargs):
        self.config = config
        self.device = device
        checkpoint = torch.load(config["init"], map_location=self.device)
        self.args = checkpoint["args"]
        expected_args = inspect.getfullargspec(NISQA).args
        model_args = {k: v for k, v in self.args.items() if k in expected_args}
        self.model = NISQA(**model_args)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    def train(self, exp_folder, datafiles):
        raise NotImplementedError

    def eval_dir(self, exp_folder: str, datafile: str, *args) -> None:
        """
        Run all samples in the datafile through the NISQA model and dump their MOS.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
        """

        # create the dump folder and the file that stores the EERs
        dump_folder = os.path.join(exp_folder, "eval", "nisqa")
        os.makedirs(dump_folder, exist_ok=True)
        fname = os.path.splitext(os.path.basename(datafile))[0]

        # run the evaluation with the anonymized data
        dataset, fnames = create_dataset(datafile, self.args)
        preds, _ = predict_mos(
            self.model,
            dataset,
            self.config.batch_size,
            self.device,
            num_workers=self.config.num_workers,
        )

        # write the predictions to a file
        with open(os.path.join(dump_folder, f"{fname}.txt"), "w") as f:
            f.write("path mos\n")
            for i in range(len(preds)):
                # round to 3 decimals
                f.write(f"{fnames[i]} {preds[i].item():.2f}\n")

        analyse_results(dump_folder, datafile, preds, analyse_func, headers_func)


def create_dataset(datafile: str, args: dict) -> tuple[SpeechQualityDataset, list[str]]:
    """
    Adaptation from NISQA._loadDatasetsCSVpredict to our datafiles.

    Args:
        datafile: path to the datafile
        args: the args from the NISQA model

    Returns:
        the dataset used to pass the data through the model
        the filenames, needed to match the predictions to the original files.
    """
    col = "path"
    fnames = [json.loads(line)[col] for line in open(datafile)]
    dataset = SpeechQualityDataset(
        pd.DataFrame(fnames, columns=[col]),
        df_con=None,
        # data_dir=args["data_dir"],
        filename_column=col,
        mos_column="predict_only",
        seg_length=args["ms_seg_length"],
        max_length=args["ms_max_segments"],
        to_memory=False,
        to_memory_workers=None,
        seg_hop_length=args["ms_seg_hop_length"],
        transform=None,
        ms_n_fft=args["ms_n_fft"],
        ms_hop_length=args["ms_hop_length"],
        ms_win_length=args["ms_win_length"],
        ms_n_mels=args["ms_n_mels"],
        ms_sr=args["ms_sr"],
        ms_fmax=args["ms_fmax"],
        # ms_channel=None,
        # double_ended=args["double_ended"],
        # dim=args["dim"],
        # filename_column_ref=args["csv_ref"],
    )
    return dataset, fnames

import os
import logging

import torch
from torch.nn.utils.rnn import pad_sequence
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor
import numpy as np

from spkanon_eval.evaluate import SAMPLE_RATE
from spkanon_eval.evaluation.ser.model_utils import EmotionModel
from spkanon_eval.evaluation.ser.analysis_utils import analyse_func, headers_func
from spkanon_eval.evaluation.analysis import analyse_results
from spkanon_eval.datamodules import eval_dataloader
from spkanon_eval.component_definitions import InferComponent, EvalComponent


LOGGER = logging.getLogger("progress")


class EmotionEvaluator(InferComponent, EvalComponent):
    def __init__(self, config, device, **kwargs):
        self.config = config
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(config.init)
        self.model = EmotionModel.from_pretrained(config.init).to(device)
        self.model.eval()

        # prepare the config for the dataloader
        self.config.data.config.batch_size = config.batch_size
        self.config.data.config.sample_rate = SAMPLE_RATE

    def to(self, device):
        self.device = device
        self.model.to(device)

    def run(self, batch):
        """
        Return the emotion dimensions for the batch. Each sample is given a 3D vector
        that defines its arousal, dominance and valence.
        """
        y = self.processor(batch[0].cpu().numpy(), sampling_rate=SAMPLE_RATE)
        y = torch.tensor(y["input_values"]).to(self.device)
        return self.model(y)

    def train(self, exp_folder, datafiles):
        raise NotImplementedError

    def eval_dir(self, exp_folder: str, datafile: str, is_baseline: bool) -> None:
        """
        Compute the cosine similarity between the emotion embedding of the original
        and the anonymized samples. The embeddings are computed with the pretrained
        Wav2Vec2 model.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
            is_baseline: whether the baseline is being evaluated
        """
        eval_dir = "ser-audeering-w2v"
        if is_baseline:
            eval_dir += "-baseline"
        dump_folder = os.path.join(exp_folder, "eval", eval_dir)

        os.makedirs(dump_folder, exist_ok=True)
        root_folder = os.path.join(exp_folder, "results")

        # init the lists that will store the results to be analysed later
        dims = {"arousal": [], "dominance": [], "valence": []}
        emb_similarities = list()

        # create the dump file for this datafile
        dump_file = os.path.join(dump_folder, os.path.basename(datafile))
        with open(dump_file, "w") as f:
            if is_baseline:
                f.write(f"path {' '.join(dims)}\n")
            else:
                f.write("path similarity ")
                f.write(" ".join(dims) + " ")
                f.write(" ".join([f"{dim}_diff" for dim in dims]) + "\n")

        for _, batch, sample_data in eval_dataloader(
            self.config.data.config, datafile, self
        ):
            # compute the emotion dimensions for the batch
            embs_y, dims_y = self.run(batch)

            # if we are evaluating the baseline, dump the dims and continue
            if is_baseline:
                with open(dump_file, "a") as f:
                    for i in range(len(sample_data)):
                        f.write(f"{sample_data[i]['path']} ")
                        for j in range(len(dims)):
                            f.write(f"{dims_y[i][j]} ")
                        f.write("\n")
                continue

            # compute the emotion dimensions of the original audio
            audios_x = [
                torchaudio.load(
                    s["path"].replace(f"./{exp_folder}/results", root_folder)
                )
                for s in sample_data
            ]
            resampled_x = list()
            for audio, sr in audios_x:
                if sr != SAMPLE_RATE:
                    audio = Resample(sr, SAMPLE_RATE)(audio)
                resampled_x.append(audio)
            batch_x = [pad_sequence(resampled_x, batch_first=True)]
            if batch_x[0].shape[0] == 1:
                batch_x[0] = batch_x[0].squeeze(0)
            embs_x, dims_x = self.run(batch_x)

            # compare the emotion content of the original and the anonymized audio
            similarity = torch.nn.functional.cosine_similarity(embs_x, embs_y)
            dim_diff = dims_x - dims_y

            # write the results to the dump file
            with open(dump_file, "a") as f:
                for i in range(len(sample_data)):
                    # write the audio filepath and the embedding cosine similarity
                    f.write(f"{sample_data[i]['path']} {similarity[i]} ")
                    # write the predicted emotion dimensions for the anonymized audio
                    for j in range(len(dims)):
                        f.write(f"{dims_y[i][j]} ")
                    # write the difference between the original and the anonymized audio
                    for j in range(len(dims)):
                        f.write(f"{dim_diff[i][j]} ")
                    f.write("\n")

            # store this batch's results for later analysis
            emb_similarities += similarity.tolist()
            for i, dim in enumerate(dims):
                dims[dim] += dim_diff[:, i].tolist()

        if not is_baseline:
            data = [
                np.array(emb_similarities),
                {dim: np.array(dims[dim]) for dim in dims},
            ]
            analyse_results(dump_folder, datafile, data, analyse_func, headers_func)

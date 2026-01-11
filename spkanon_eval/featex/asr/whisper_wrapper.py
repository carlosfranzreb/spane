import os
import string
import logging

import torch
import numpy as np
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, AutoProcessor
import editdistance

from spkanon_eval.datamodules.dataloader import eval_dataloader
from spkanon_eval.evaluation.analysis import analyse_results
from spkanon_eval.featex.asr.whisper_analysis_utils import analyse_func, headers_func
from spkanon_eval.component_definitions import InferComponent, EvalComponent
from spkanon_eval.datamodules import AudioBatch

LOGGER = logging.getLogger("progress")


class Whisper(InferComponent, EvalComponent):
    def __init__(self, config, device, **kwargs):
        self.device = device
        self.config = config
        model_id = f"openai/whisper-{config.size}"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def run(self, batch: AudioBatch) -> list:
        """
        Uses HF's transformers implementation for batch transcription of long-form
        audio. See <https://github.com/huggingface/transformers/pull/27658>
        """

        # create input for Whisper
        inputs = [
            audio[: batch.lens[idx]]
            for idx, audio in enumerate(batch.audios.cpu().numpy())
        ]
        inputs = self.processor(
            inputs,
            return_tensors="pt",
            truncation=False,
            padding="longest",
            return_attention_mask=True,
            sampling_rate=16_000,
        )
        inputs = inputs.to(self.device, torch.float16)

        # run Whisper
        result = self.model.generate(
            **inputs,
            language=self.config.language,
            condition_on_prev_tokens=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            logprob_threshold=-1.0,
            compression_ratio_threshold=1.35,
            return_timestamps=False,
        )
        decoded = self.processor.batch_decode(result, skip_special_tokens=True)

        return decoded

    def train(self, exp_folder: str) -> None:
        raise NotImplementedError("Whisper wrapper does not support training")

    def eval_dir(self, exp_folder: str, datafile: str, *args) -> None:
        """
        Compute the transcription of each sample and its WER. Dump both into a file
        within the current experiment folder. Keep track of the average WER of each
        datafile and each speaker characteristic (age, gender, etc). Dump these
        averages as well.

        Args:
            exp_folder: path to the experiment folder
            datafile: datafile to evaluate
        """
        LOGGER.info("Computing WER of eval data with dataloader")
        dump_folder = os.path.join(exp_folder, "eval", f"whisper-{self.config.size}")
        os.makedirs(dump_folder, exist_ok=True)
        stats = {"n_edits": list(), "n_words_ref": list()}

        # define the dump file and write the headers
        dump_file = os.path.join(dump_folder, os.path.basename(datafile))
        with open(dump_file, "w", encoding="utf-8") as f:
            f.write("path n_edits n_words_ref wer text\n")

        for batch in tqdm(eval_dataloader(self.config.data.config, datafile, self)):
            texts_pred = self.run(batch)  # compute the transcriptions for the batch
            for i, text_pred in enumerate(texts_pred):  # iterate through the batch
                # compute the WER for the current sample
                audiofile = batch.metadata[i]["path"]
                text_ref = batch.metadata[i]["text"]
                n_edits, n_words, wer = compute_edits(text_pred, text_ref)
                # if wer could not be computed, skip
                if n_words == 0:
                    LOGGER.warning(
                        f"Reference text of {audiofile} has no words; WER = 0"
                    )
                # dump the results for this sample into the datafile
                with open(dump_file, "a", encoding="utf-8") as f:
                    f.write(f"{audiofile} {n_edits} {n_words} {wer} {text_pred}\n")
                # update datafile stats
                stats["n_edits"].append(n_edits)
                stats["n_words_ref"].append(n_words)

        analyse_results(
            dump_folder,
            datafile,
            [np.array(stats["n_edits"]), np.array(stats["n_words_ref"])],
            analyse_func,
            headers_func,
        )

    def to(self, device):
        """Implementation of PyTorch's `to()` method to set the device."""
        self.model.to(device)
        self.device = device


def compute_edits(text_pred, text_ref):
    """
    Normalize the texts, split them into words and compute the Levenhstein
    distance between the lists of words. Return the edit distance, the number of words
    of `text_ref` (which we assume is the reference) and the WER.
    The max. WER is 1. If the reference text has no words, return all zeros so that the
    analysis may be run.
    """
    text_pred = normalize_text(text_pred)
    text_ref = normalize_text(text_ref)
    n_edits = editdistance.eval(text_pred, text_ref)
    if len(text_ref) > 0:
        n_words_ref = len(text_ref)
        if n_edits > n_words_ref:
            n_edits = n_words_ref
        return n_edits, n_words_ref, round(n_edits / n_words_ref, 2)
    else:
        return 0, 0, 0


def normalize_text(text):
    """
    Normalize the text by removing the punctuation and making it lowercase.
    Split the text into words and return the list of words.
    """
    return text.translate(str.maketrans("", "", string.punctuation)).lower().split()


def dump_averages(out_dir, datafile, subset, data):
    """
    Add the data to the appropriate file. The file is determined by the subset and is
    created under `out_dir` if it does not exist. The data is added to the file as a
    new line.
    """
    # define the columns used for this subset; `totals` has no subset column
    cols = ["datafile", "n_edits", "n_words_ref", "wer"]
    if subset != "totals":
        cols.insert(1, subset)

    # define the filename and create it with headers if needed
    fname = os.path.join(out_dir, f"avg_wers-{subset}.txt")
    if not os.path.exists(fname):
        with open(fname, "w") as f:
            f.write(" ".join(cols) + "\n")

    # define a subroutine that writes data to the file
    def write_data(data):
        data["datafile"] = datafile
        data["wer"] = round(data["n_edits"] / data["n_words_ref"], 2)
        with open(fname, "a") as f:
            f.write(" ".join([str(data[col]) for col in cols]) + "\n")

    # write the data for this subset; for speaker chars., iterate over the values
    if subset == "totals":
        write_data(data)
    else:
        for subset_val, val_data in data.items():
            val_data[subset] = subset_val
            write_data(val_data)

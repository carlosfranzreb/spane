import os
import json
import logging
from typing import TextIO

from torch.cuda import OutOfMemoryError, empty_cache
import torchaudio
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from spkanon_eval.datamodules import eval_dataloader
from spkanon_eval.anonymizer import Anonymizer


LOGGER = logging.getLogger("progress")


def infer(exp_folder: str, df_name: str, model: Anonymizer, config: DictConfig) -> str:
    """
    - If the output datafile already exists, return its path.
    - Run each recording through the model and store the resulting audiofiles in
        `{exp_folder}/results/{df_name}`.
    - Iterate through the audiofiles of `exp_folder/data/{df_name}.txt` with the
        eval_dataloader, which returns waveforms and info about each sample.
    - Once the batch is anonymized, each sample is trimmed and saved to the
        corresponding path.
    - Anonymized samples are saved with their targets to the new datafile
        `{exp_folder}/data/anon_{df_name}.txt`.

    Args:
        exp_folder: path to the experiment folder
        df_name: name of the datafile, without the directory or the extension (e.g.
        "eval"). The corresponding datafile is assumed to be in `{exp_folder}/data`.
        model: the anonymizer model
        config: the config object, as defined in the documentation.

    Returns:
        The path to the datafile with the anonymized samples and their targets.
    """

    datafile = os.path.join(exp_folder, "data", f"{df_name}.txt")
    anon_datafile = os.path.join(exp_folder, "data", f"anon_{df_name}.txt")
    if os.path.exists(anon_datafile):
        LOGGER.warning(f"Anonymized data for {df_name} already exists")
        return anon_datafile

    writer = open(anon_datafile, "w")
    dump_dir = os.path.join(exp_folder, "results", df_name)
    data_cfg = config.data.config
    sample_rate = config.data.config.sample_rate

    def infer_batch(batch: list, data: list):
        audio_anon, n_samples, target = model.forward(batch, data)
        for idx in range(len(audio_anon)):
            data[idx]["path"] = data[idx]["path"].replace(
                data_cfg.root_folder, dump_dir
            )
            format = os.path.splitext(data[idx]["path"])[1][1:]
            os.makedirs(os.path.split(data[idx]["path"])[0], exist_ok=True)
            torchaudio.save(
                data[idx]["path"],
                audio_anon[idx, :, : n_samples[idx]].cpu().detach(),
                sample_rate,
                format=format,
            )
            data[idx]["duration"] = round(n_samples[idx].item() / sample_rate, 3)
            data[idx]["target"] = target[idx].item()
            writer.write(json.dumps(data[idx]) + "\n")

    def oom_handler(batch: list, data: list):
        try:
            infer_batch(batch, data)
        except OutOfMemoryError as error:
            batch_size = batch[0].shape[0]
            if batch_size == 1:
                LOGGER.error("Out of memory with batch size 1.")
                raise error
            else:
                LOGGER.warning("Out of memory, retrying with half batch sizes.")
                empty_cache()
                half_idx = batch_size // 2
                oom_handler([b[:half_idx] for b in batch], data[:half_idx])
                oom_handler([b[half_idx:] for b in batch], data[half_idx:])

    for _, batch, data in tqdm(eval_dataloader(data_cfg, datafile, model.device)):
        oom_handler(batch, data)

    writer.close()

    # sort the datafile by duration
    objs = [json.loads(line) for line in open(anon_datafile)]
    objs.sort(key=lambda x: x["duration"], reverse=True)
    with open(anon_datafile, "w") as f:
        for obj in objs:
            f.write(json.dumps(obj) + "\n")

    return anon_datafile

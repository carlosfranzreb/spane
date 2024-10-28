import os
import json
import logging
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torchaudio
from omegaconf import OmegaConf
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset

from spkanon_eval.datamodules.dataset import SpeakerIdDataset

LOGGER = logging.getLogger("progress")


def setup_dataloader(config: OmegaConf, datafile: str) -> DataLoader:
    """
    Create a dataloader with the SpeakerIdDataset.
    """

    LOGGER.info(f"Creating dataloader for {datafile}")
    LOGGER.info(f"\tSample rate: {config.sample_rate}")
    LOGGER.info(f"\tNum. workers: {config.num_workers}")

    fname = os.path.splitext(os.path.basename(datafile))[0]
    fname = fname.replace("anon_", "")
    if "trials" in fname or "enrolls" in fname:
        fname = fname.split("_")[0]

    return DataLoader(
        dataset=SpeakerIdDataset(
            datafile, config.sample_rate, config.chunk_sizes[fname]
        ),
        num_workers=config.num_workers,
        batch_size=None,
    )


def eval_dataloader(
    config: OmegaConf, datafile: str, device: str
) -> Iterable[str, list[Tensor], dict[str, str]]:
    """
    This function is called by evaluation and inference scripts. It is an
    iterator over the batches and other sample info in the given manifest.

    - The data is not shuffled, so it can be mapped to the audio file paths, which
        they require to generate their results/reports.
    - Return all additional data found in the manifest file, if any. This can be the
        gender of the speaker, for example.
    """
    LOGGER.info(f"Creating eval. DL for `{datafile}`")

    # initialize the dataloader and the iterator object for the sample data
    dl = setup_dataloader(config, datafile)
    data_iter = data_iterator(datafile)

    # iterate over the batches in the dataloader
    for batch in dl:
        batch = [b.to(device) for b in batch]
        data = list()  # additional data to be returned
        # read as much `data` as there are samples in the batch
        while len(data) < batch[0].shape[0]:
            data.append(next(data_iter))
        # yield the batch, the datafile and the additional data
        yield datafile, batch, data


def data_iterator(datafile: str) -> Iterable[dict]:
    """Iterate over the JSON objects in the given datafile."""
    with open(datafile) as f:
        for line in f:
            yield json.loads(line)

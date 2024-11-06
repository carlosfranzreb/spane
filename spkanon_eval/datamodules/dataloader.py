import json
import logging
from collections.abc import Iterable

from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from spkanon_eval.datamodules import SpeakerIdDataset, BatchSizeCalculator

LOGGER = logging.getLogger("progress")
bs_calculator = BatchSizeCalculator()


def setup_dataloader(
    model, config: OmegaConf, datafile: str, max_ratio: float = 0.8
) -> DataLoader:
    """
    Create a dataloader with the SpeakerIdDataset.
    """

    LOGGER.info(f"Creating dataloader for {datafile}")
    LOGGER.info(f"\tModel: {model.__class__.__name__}")
    LOGGER.info(f"\tSample rate: {config.sample_rate}")
    LOGGER.info(f"\tNum. workers: {config.num_workers}")

    chunk_sizes = bs_calculator.calculate(
        datafile, model, config.sample_rate, max_ratio
    )
    return DataLoader(
        dataset=SpeakerIdDataset(datafile, config.sample_rate, chunk_sizes),
        num_workers=config.num_workers,
        batch_size=None,
    )


def eval_dataloader(
    config: OmegaConf, datafile: str, model, max_ratio: float = 0.8
) -> Iterable[str, list[Tensor], dict[str, str]]:
    """
    This function is called by evaluation and inference scripts. It is an
    iterator over the batches and other sample info in the given manifest.

    - The data is not shuffled, so it can be mapped to the audio file paths, which
        they require to generate their results/reports.
    - Return all additional data found in the manifest file, e.g. gender, speaker_id.

    Args:
        config: the configuration object
        datafile: the path to the manifest file
        model: the model to evaluate
        max_ratio: the ratio of the GPU memory to use (see `BatchSizeCalculator`)
    """
    LOGGER.info(f"Creating eval. DL for `{datafile}`")

    # initialize the dataloader and the iterator object for the sample data
    dl = setup_dataloader(model, config, datafile, max_ratio)
    data_iter = data_iterator(datafile)

    # iterate over the batches in the dataloader
    for batch in dl:
        batch = [b.to(model.device) for b in batch]
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

"""
Class that computes the max. batch size that fits into GPU memory.

It stores previous computations to avoid recomputing the same chunk size for the 
same model and sample rate.
"""

import logging
import json
from math import ceil

import torch
from tqdm import tqdm

LOGGER = logging.getLogger("progress")
SIZE_INCREASE = 8  # min. increase of batch size


class BatchSizeCalculator:
    def __init__(self, n_chunks: int = 5):
        """
        `self.chunks` stores already computed chunk sizes per model and sample rate.
        """
        self.n_chunks = n_chunks
        self.chunks = dict()

    def calculate(
        self, datafile: str, model, sample_rate: int, max_ratio: float
    ) -> dict:
        """
        Compute the chunk sizes for the given datafile. The chunk size determines the
        batch size depending on its maximum duration, to maximize GPU memory usage. The
        datafile is expected to contain samples sorted by duration in descending order,
        as produced by the `prepare_datafile` function.

        If a similar computation (+ 1s) has already been done for the same model and
        sample rate, it is returned from memory instead of recomputed.

        Args:
            datafile: path to the datafile with sorted samples
            model: the model for which the chunk sizes are computed. It must have either
                a `forward` or `run` method.
            sample_rate: the sample rate of the audio files in the datafile
            max_ratio: the ratio of the GPU memory to use.


        Returns:
            A dictionary mapping the maximum duration of a batch to the max. number of
            samples of that duration that can fit in GPU memory.
        """
        LOGGER.info(
            f"Computing chunk sizes for {datafile} and {model.__class__.__name__}"
        )

        # read the first and ~last lines of the datafile to get the min and max duration
        with open(datafile) as f:
            lines = f.readlines()
        min_line = -10 if len(lines) > self.n_chunks * 10 else -1
        min_dur = float(json.loads(lines[min_line])["duration"])
        max_dur = float(json.loads(lines[0])["duration"])

        if model.device == "cpu":
            LOGGER.warning("\tModel is on CPU. Skipping chunk size computation.")
            return {ceil(max_dur): 2}

        model_key = (id(model), sample_rate)
        first_time = model_key not in self.chunks
        if first_time:
            self.chunks[model_key] = dict()

        total_memory = torch.cuda.get_device_properties(0).total_memory
        LOGGER.info(f"\tTarget GPU memory usage: {(total_memory / 1024 ** 2):.2f} MB")

        out_sizes = dict()
        batch_size = 1
        for chunk_max_dur in tqdm(
            torch.linspace(max_dur, min_dur, self.n_chunks + 1)[:-1]
        ):

            # reset batch size to the last value that worked for the larger chunk size
            if len(out_sizes) > 0:
                batch_size = list(out_sizes.values())[-1] + SIZE_INCREASE

            # check if the chunk size has already been computed
            if not first_time:
                found = False
                for dur, bs in self.chunks[model_key].items():
                    diff = dur - chunk_max_dur
                    if diff >= 0 and diff <= 1:
                        out_sizes[ceil(dur)] = bs
                        found = True
                        break
                if found:
                    continue

            # compute the batch size for the current max. duration
            chunk_max_dur = torch.ceil(chunk_max_dur).item()
            n_samples = int(chunk_max_dur * sample_rate)
            while True:
                batch = [
                    torch.randn([batch_size, n_samples], device=model.device),
                    torch.randint(10, [batch_size], device=model.device),
                    torch.ones(batch_size, device=model.device, dtype=torch.int32)
                    * n_samples,
                ]
                torch.cuda.reset_peak_memory_stats()
                try:
                    if hasattr(model, "forward"):
                        data = [
                            {"speaker_id": val.item(), "gender": True}
                            for val in batch[1]
                        ]
                        model.forward(batch, data)
                    else:
                        model.run(batch)

                    out_sizes[chunk_max_dur] = batch_size
                    self.chunks[model_key][chunk_max_dur] = batch_size
                    max_usage = torch.cuda.max_memory_allocated()
                    batch_size = max(
                        batch_size + SIZE_INCREASE,
                        int(batch_size * (total_memory / max_usage)),
                    )

                except torch.cuda.OutOfMemoryError:
                    break
                except RuntimeError as error:
                    if "must fit into 32-bit index math" in str(error):
                        break
                    else:
                        LOGGER.error(error)
                        raise error

        out_sizes = {k: int(v * max_ratio) for k, v in out_sizes.items()}
        LOGGER.info(f"\tComputed chunk sizes: {out_sizes}")
        return out_sizes

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


class BatchSizeCalculator:
    def __init__(self, n_chunks: int = 5):
        """
        `self.chunks` stores already computed chunk sizes per model and sample rate.
        """
        self.n_chunks = n_chunks
        self.chunks = dict()

    def calculate(self, datafile: str, model, sample_rate: int) -> dict:
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
            n_chunks: the number of chunks to compute

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
            return {max_dur: 1}

        total_memory = torch.cuda.get_device_properties(0).total_memory
        LOGGER.info(f"\tTarget GPU memory usage: {(total_memory / 1024 ** 2):.2f} MB")
        out_sizes = dict()
        batch_size = 1
        for chunk_max_dur in tqdm(
            torch.linspace(max_dur, min_dur, self.n_chunks + 1)[:-1]
        ):

            # check if the chunk size has already been computed
            if (id(model), sample_rate) in self.chunks:
                found = False
                for dur, bs in self.chunks[(id(model), sample_rate)].items():
                    diff = dur - chunk_max_dur
                    if diff >= 0 and diff <= 1:
                        out_sizes[ceil(dur)] = bs
                        found = True
                        break
                if found:
                    continue
            else:
                self.chunks[(id(model), sample_rate)] = dict()

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
                    self.chunks[(id(model), sample_rate)][chunk_max_dur] = batch_size
                    max_usage = torch.cuda.max_memory_allocated()
                    batch_size = max(
                        batch_size + 4,
                        int(batch_size * (total_memory / max_usage) * 0.8),
                    )

                except torch.cuda.OutOfMemoryError:
                    break
                except RuntimeError as error:
                    if "must fit into 32-bit index math" in str(error):
                        break
                    else:
                        LOGGER.error(error)
                        raise error

        LOGGER.info(f"\tComputed chunk sizes: {out_sizes}")
        return out_sizes

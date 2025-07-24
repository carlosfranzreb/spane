import os
import logging
import time
import sys

import torchaudio
import torch
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from spkanon_eval.component_definitions import EvalComponent
from spkanon_eval.datamodules.batch_size_calculator import map_audio_to_dur
from spkanon_eval.anonymizer import Anonymizer

LOGGER = logging.getLogger("progress")


class PerformanceEvaluator(EvalComponent):
    def __init__(self, config: DictConfig, device: str, model: Anonymizer):
        self.config = config
        self.device = device
        self.model = model

    def train(self, exp_folder: str) -> None:
        raise NotImplementedError("Performance evaluation does not require training")

    def eval_dir(self, exp_folder: str, *args) -> None:
        """
        Measure the inference time on GPU and CPU.

        Args:
            exp_folder: path to the experiment folder
        """

        dump_folder = os.path.join(exp_folder, "eval", "performance")
        os.makedirs(dump_folder, exist_ok=True)

        # evaluate the model's performance on the GPU if possible
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA is not available, skipping GPU evaluation")
        else:
            device = "cuda"
            os.system(f"nvidia-smi > {os.path.join(dump_folder, 'gpu_specs.txt')}")
            self.model.to(device)
            LOGGER.info("Evaluating the GPU inference time")
            inference_time(
                self.model,
                os.path.join(dump_folder, f"{device}_inference.txt"),
                run_gpu,
                self.config.repetitions,
                self.config.durations,
                self.config.data.config.sample_rate_in,
            )

        # write the model's CPU specs to the experiment folder
        operating_system = sys.platform
        f_specs = os.path.join(dump_folder, "cpu_specs.txt")
        if operating_system == "darwin":
            os.system(f"sysctl -a | grep machdep.cpu > {f_specs}")
        elif operating_system == "linux":
            os.system(f"lscpu > {f_specs}")
        else:
            LOGGER.error(f"Can't get the CPU specs of OS {operating_system}")

        # evaluate the model's performance on the CPU
        device = "cpu"
        self.model.to(device)
        LOGGER.info("Evaluating the CPU inference time")
        inference_time(
            self.model,
            os.path.join(dump_folder, f"{device}_inference.txt"),
            run_cpu,
            self.config.repetitions,
            self.config.durations,
            self.config.data.config.sample_rate_in,
        )

        # reset the model to its original device
        self.model.to(self.device)


def inference_time(
    model: Anonymizer,
    dump_file: str,
    func,
    repetitions: int,
    durations: list[int],
    sample_rate: int,
):
    """
    Computes the mean and std inference time of the model on `device` for inputs of
    different durations, writing the results to `dump_file`.

    - `model` is a PyTorch model with a `forward` method that receives a batch from a
        dataloader (signal, label) and a target ID for each sample in the batch.
    - `dump_file` (str): path to the file where the results are written.
    - `func`: function that runs the experiment, timing the duration of each forward
            pass and returning them as a numpy array.
    - `repetitions` (int): number of times the batch is passed to the model.
    - `durations` (list of ints): durations (in seconds) for which to perform the
        experiment.
    - `sample_rate` (int): sample rate of the model.
    """

    # write the headers to the dump file
    with open(dump_file, "w") as f:
        f.write("input_duration inference_mean inference_std\n")

    audio = torchaudio.load(
        "spkanon_eval/tests/data/LibriSpeech/dev-clean-2/2412/153948/2412-153948-0000.flac"
    )[0].squeeze()

    for duration in durations:
        n_samples = int(sample_rate * duration)
        map_audio_to_dur(audio, n_samples)
        timings = func(model, audio, 1, repetitions)

        # dump the mean and std of the inference time
        with open(dump_file, "a") as f:
            f.write(f"{duration}")
            f.write(f" {np.round(np.mean(timings), 3)}")
            f.write(f" {np.round(np.std(timings), 3)}\n")


def run_gpu(
    model: Anonymizer, audio: torch.Tensor, batch_size: int, repetitions: int
) -> np.ndarray:
    """
    Runs a given model on GPU with the specified input size for the specified number of
    repetitions.

    Args:
        model: The anonymizer model to run on GPU.
        audio: real audio to use in the batch, of shape (n_samples,)
        batch_size: no. of samples to include in the batch
        repetitions: no. of times to repeat the model execution.

    Returns:
        numpy.ndarray: A 2D array containing the execution timings of the model in
        seconds. The shape of the array is (repetitions, 1).

    Example:
        # Create a model and run it on GPU with a batch size of 32 for 10 repetitions
        model = MyModel()
        timings = run_gpu(model, 10, (32, 128))
        print(timings)
    """

    with torch.no_grad():

        # batch comprises a signal, a speaker label and the audio length
        audio_batch = audio.unsqueeze(0).repeat(batch_size, 1)
        batch = [
            audio_batch.to(model.device),
            torch.randint(10, [batch_size], device=model.device),
            torch.ones(batch_size, device=model.device, dtype=torch.int32)
            * audio.shape[0],
        ]
        data = [{"speaker_id": val.item(), "gender": True} for val in batch[1]]

        # warm-up
        for _ in range(10):
            model.forward(batch, data)

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = np.zeros((repetitions, 1))

        for idx in tqdm(range(repetitions)):
            starter.record()
            model.forward(batch, data)
            ender.record()
            torch.cuda.synchronize()
            timings[idx] = starter.elapsed_time(ender) / 1000

    return timings


def run_cpu(
    model: Anonymizer, audio: torch.Tensor, batch_size: int, repetitions: int
) -> np.ndarray:
    """
    Runs a given model on CPU with the specified input size for the specified number of
    repetitions.

    Args:
        model: The anonymizer model to run on GPU.
        audio: real audio to use in the batch, of shape (n_samples,)
        batch_size: no. of samples to include in the batch
        repetitions: no. of times to repeat the model execution.

    Returns:
        numpy.ndarray: A 2D array containing the execution timings of the model in
        seconds. The shape of the array is (repetitions, 1).

    Example:
        # Create a model and run it on CPU with a batch size of 32 for 10 repetitions
        model = MyModel()
        timings = run_cpu(model, 10, (32, 128))
        print(timings)
    """

    # batch comprises a signal, a speaker label and the audio length
    audio_batch = audio.unsqueeze(0).repeat(batch_size, 1)
    batch = [
        audio_batch.to(model.device),
        torch.randint(10, [batch_size], device=model.device),
        torch.ones(batch_size, device=model.device, dtype=torch.int32) * audio.shape[0],
    ]
    data = [{"speaker_id": val.item(), "gender": True} for val in batch[1]]

    # warm-up
    for _ in range(10):
        model.forward(batch, data)

    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for i in range(repetitions):
            start_time = time.time()
            model.forward(batch, data)
            timings[i] = time.time() - start_time

    return timings

import os
import logging
import time
import sys

import torch
import numpy as np

from spkanon_eval.component_definitions import EvalComponent


LOGGER = logging.getLogger("progress")


class PerformanceEvaluator(EvalComponent):
    def __init__(self, config, device, model):
        self.config = config
        self.device = device
        self.model = model

    def train(self, exp_folder: str) -> None:
        raise NotImplementedError("Performance evaluation does not require training")

    def eval_dir(self, exp_folder: str, *args) -> None:
        """
        Measure the inference time and throughput of the model, on GPU and CPU.

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
                self.config.sample_rate,
            )
            LOGGER.info("Evaluating the GPU throughput")
            gpu_throughput(
                self.model,
                os.path.join(dump_folder, "gpu_throughput.txt"),
                self.config.repetitions,
                self.config.durations,
                self.config.sample_rate,
            )

        # write the model's CPU specs to the experiment folder
        operative_system = sys.platform
        f_specs = os.path.join(dump_folder, "cpu_specs.txt")
        if operative_system == "darwin":
            os.system(f"sysctl -a | grep machdep.cpu > {f_specs}")
        elif operative_system == "linux":
            os.system(f"lscpu > {f_specs}")
        else:
            LOGGER.error(f"Can't get the CPU specs of OS {operative_system}")

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
            self.config.sample_rate,
        )

        # reset the model to its original device
        self.model.to(self.device)


def inference_time(model, dump_file, func, repetitions, durations, sample_rate):
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

    for duration in durations:
        n_frames = int(sample_rate * duration)
        timings = func(model, repetitions, (1, n_frames))

        # dump the mean and std of the inference time
        with open(dump_file, "a") as f:
            f.write(f"{duration}")
            f.write(f" {np.round(np.mean(timings), 3)}")
            f.write(f" {np.round(np.std(timings), 3)}\n")


def gpu_throughput(model, dump_file, repetitions, durations, sample_rate):
    """
    Computes the throughput of the model on CPU for inputs of
    different durations, writing the results to `dump_file`.

    - `model` is a PyTorch model with a `forward` method that receives a batch from a
        dataloader (signal, label) and a target ID for each sample in the batch.
    - `dump_file` (str): path to the file where the results are written.
    - `repetitions` (int): number of times the batch is passed to the model.
    - `durations` (list of ints): durations (in seconds) for which to perform the
        experiment.
    - `sample_rate` (int): sample rate of the model.
    """

    # write the headers to the dump file
    with open(dump_file, "w") as f:
        f.write("input_duration max_batch_size throughput\n")

    for duration in durations:
        n_frames = int(sample_rate * duration)
        batch_size = max_batch_size(model, n_frames, "cuda")
        if batch_size == 0:
            throughput = "OOM_error"
        else:
            try:
                timings = run_gpu(model, repetitions, (batch_size, n_frames))
                total_time = np.sum(timings)
                throughput = round((repetitions * batch_size) / total_time, 3)
            except RuntimeError as err:
                if "out of memory" in str(err):
                    throughput = "OOM_error"
                    break
                else:
                    raise err

        # dump the batch size and the throughput
        with open(dump_file, "a") as f:
            f.write(f"{duration} {batch_size} {throughput}\n")


def run_gpu(model, repetitions, size):
    """
    Runs a given model on GPU with the specified input size for the specified number of
    repetitions.

    Args:
        model (torch.nn.Module): The model to run on GPU.
        repetitions (int): The number of times to repeat the model execution.
        size (tuple): The size of the input batch. Should be a tuple of integers.

    Returns:
        numpy.ndarray: A 2D array containing the execution timings of the model in
        seconds. The shape of the array is (repetitions, 1).

    Example:
        # Create a model and run it on GPU with a batch size of 32 for 10 repetitions
        model = MyModel()
        timings = run_gpu(model, 10, (32, 128))
        print(timings)
    """

    # batch comprises a signal, a speaker label and the audio length
    batch = [
        torch.randn(size, dtype=torch.float).to("cuda"),
        torch.ones(size[0]),
        torch.ones(size[0], dtype=torch.int32) * size[-1],
    ]
    data = [{"speaker_id": 1} for _ in range(size[0])]

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    with torch.no_grad():
        # GPU-WARM-UP
        for _ in range(10):
            model.forward(batch, data)

        for i in range(repetitions):
            starter.record()
            model.forward(batch, data)
            ender.record()
            torch.cuda.synchronize()
            timings[i] = starter.elapsed_time(ender) / 1000

    return timings


def run_cpu(model, repetitions, size):
    """
    Runs a given model on CPU with the specified input size for the specified number of
    repetitions.

    Args:
        model (torch.nn.Module): The model to run on CPU.
        repetitions (int): The number of times to repeat the model execution.
        size (tuple): The size of the input batch. Should be a tuple of integers.

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
    batch = [
        torch.randn(size, dtype=torch.float).to("cpu"),
        torch.ones(size[0]),
        torch.ones(size[0], dtype=torch.int32) * size[-1],
    ]
    data = [{"speaker_id": 1} for _ in range(size[0])]

    timings = np.zeros((repetitions, 1))
    with torch.no_grad():
        for i in range(repetitions):
            start_time = time.time()
            model.forward(batch, data)
            timings[i] = time.time() - start_time

    return timings


def max_batch_size(model, input_size, device):
    """
    Determines the maximum batch size that can fit in the GPU memory for a given model
    and input size.

    Args:
        model (torch.nn.Module): The model to evaluate batch size for.
        size (int): The size of the input tensor (excluding batch dimension).
        device (str or torch.device): The device to run the evaluation on (e.g.,
            "cuda", "cuda:0", torch.device("cuda")).

    Returns:
        int: The maximum batch size that can fit in the GPU memory.

    Raises:
        RuntimeError: If an error occurs during the evaluation.

    Example:
        # Create a model and determine the maximum batch size for an input size of 128
        model = MyModel()
        max_size = max_batch_size(model, 128, "cuda")
        print(max_size)
    """

    batch_size = 1
    while batch_size > 0:
        try:
            # batch comprises a signal, a speaker label and the audio length
            size = (batch_size, input_size)
            batch = [
                torch.randn(size, dtype=torch.float).to("cuda"),
                torch.ones(size[0]),
                torch.ones(size[0], dtype=torch.int32) * size[-1],
            ]
            data = [{"speaker_id": 1} for _ in range(size[0])]
            model.forward(batch, data)
            batch_size += 1
            torch.cuda.synchronize()
        except RuntimeError as err:
            if "out of memory" in str(err):
                batch_size -= 1
                break
            else:
                raise err

    torch.cuda.synchronize()
    return batch_size

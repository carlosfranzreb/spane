"""
Create a dataset from a list of datafiles. The datafiles should be text files, each
comprising one JSON object per line. The dataset returns tuples of (audio, speaker),
where the audio is resampled to the given sampling rate.
"""

import json
import logging

import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio


LOGGER = logging.getLogger("progress")


class SpeakerIdDataset(Dataset):
    def __init__(self, datafile: str, sample_rate: int, chunk_sizes: dict) -> None:
        """
        Create a dataset from the given datafile. The datafile should be a text file,
        comprising one JSON object per line. Each JSON object should have at least the
        following keys:

        - "path": path to the audio file
        - "speaker_id": a unique integer identifying the speaker

        The dataset returns tuples of (audio, speaker, n_samples), where the audio is
        resampled to the given sampling rate. The n_samples is the number of samples in
        the audio file.

        Args:
            datafile: path to the datafile
            sample_rate: the sampling rate to resample the audio to
            chunk_sizes: a dictionary mapping the maximum duration of a batch to the
                number of samples in the batch. This is used to maximize GPU usage.
        """
        super().__init__()
        self.sample_rate = sample_rate
        self.datafile = datafile
        self.data = [json.loads(line) for line in open(datafile)]
        self.data = sorted(self.data, key=lambda x: x["duration"], reverse=True)

        # aggregate the sorted data into chunks
        data_chunks = [[self.data[0]]]
        max_dur_key = min([k for k in chunk_sizes if k >= self.data[0]["duration"]])
        for sample in self.data[1:]:
            if len(data_chunks[-1]) < chunk_sizes[max_dur_key]:
                data_chunks[-1].append(sample)
            else:
                data_chunks.append([sample])
                max_dur_key = min([k for k in chunk_sizes if k > sample["duration"]])
        self.data = data_chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, sample_idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """
        Return the `sample_idx`-th sample of the dataset. The id is the index of the
        sample when considering all the datafiles as a single dataset. The item is
        returned as a tuple of (audio, speaker, n_samples), where the audio is
        resampled to the given sampling rate.
        """
        objs = self.data[sample_idx]
        audios = pad_sequence(
            [load_audio(obj["path"], self.sample_rate) for obj in objs],
            batch_first=True,
        )
        speaker_ids = tensor([int(obj["speaker_id"]) for obj in objs])
        audio_lens = tensor([audio.shape[0] for audio in audios])
        return audios, speaker_ids, audio_lens


def load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    """
    Load the audio from the given path. If the sampling rate is different from
    given sampling rate, resample the audio. Return the waveform as a 1D tensor.
    If the audio is stereo, returns the mean across channels.
    """

    audio, sr = torchaudio.load(audio_path, normalize=True)
    if sr != sample_rate:
        audio = torchaudio.transforms.Resample(sr, sample_rate)(audio)
    audio = audio.squeeze()
    if audio.ndim > 1:
        audio = audio.mean(dim=0)
    return audio

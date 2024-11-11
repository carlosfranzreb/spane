"""
Wrapper for Speechbrain speaker recognition models. The `run` method returns speaker
embeddings.
"""

import os
import logging
import json
import csv
import random
import shutil

from speechbrain.inference.speaker import EncoderClassifier
from hyperpyyaml import load_hyperpyyaml
from omegaconf import OmegaConf
import torch

from spkanon_eval.featex.spkid.train import SpeakerBrain, prepare_dataset
from spkanon_eval.component_definitions import InferComponent


LOGGER = logging.getLogger("progress")


class SpkId(InferComponent):
    def __init__(self, config: OmegaConf, device: str) -> None:
        """Initialize the model with the given config and freeze its parameters."""
        self.sample_rate = 16000
        self.config = config
        self.device = device
        self.save_dir = os.path.join("checkpoints", config.path)
        self.model = EncoderClassifier.from_hparams(
            source=config.path, savedir=self.save_dir, run_opts={"device": device}
        )
        if config.get("emb_model_ckpt", None) is not None:
            LOGGER.info(f"Loading emb. model from {config.emb_model_ckpt}")
            state_dict = torch.load(config.emb_model_ckpt, map_location=device)
            self.model.mods.embedding_model.load_state_dict(state_dict)
        self.model.eval()

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    @torch.inference_mode()
    def run(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """
        Return speaker embeddings for the given batch of utterances.

        Args:
            batch: A list of three tensors in the following order:
            1. waveforms with shape (batch_size, n_samples)
            2. waveform lengths with shape (batch_size), as integers
            3. waveform speaker IDs with shape (batch_size), as integers

        Returns:
            A tensor containing the speaker embeddings with shape
            (batch_size, embedding_dim).
        """
        return self.model.encode_batch(
            batch[0].to(self.device), batch[2].to(self.device), True
        ).squeeze(1)

    def train(self, dump_dir: str, datafile: str, n_speakers: int) -> None:
        """
        Train this model with the given datafiles. No checkpoint will be used as a
        starting point.

        Args:
            dump_dir: Path to the folder where the model and datafiles will be saved.
            datafile: paths to the datafile used for training.
            n_speakers: Number of speakers across all datafiles, used to initialize
                the classifier.
        """
        LOGGER.info(f"Training the spkid model with datafile {datafile}")
        os.makedirs(dump_dir, exist_ok=True)
        shutil.copyfile(
            self.config.train_config, os.path.join(dump_dir, "train_config.yaml")
        )

        with open(self.config.train_config) as f:
            hparams = load_hyperpyyaml(
                f,
                overrides={
                    "output_folder": dump_dir,
                    "out_n_neurons": n_speakers,
                    "num_workers": self.config.num_workers,
                },
            )

        # create the datafiles and writers
        splits = dict()
        for split in ["train", "val"]:
            splits[split] = dict()
            splits[split]["file"] = os.path.join(dump_dir, f"{split}.csv")
            splits[split]["writer"] = open(splits[split]["file"], "w")
            splits[split]["csv_writer"] = csv.writer(
                splits[split]["writer"],
                delimiter=",",
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
            )
            splits[split]["csv_writer"].writerow(
                ["ID", "wav", "duration", "start", "stop", "spk_id", "spk_id_encoded"]
            )

        # split the data of each speaker into training and validation sets
        speaker_objs = dict()
        for line in open(datafile):
            obj = json.loads(line)
            spk = obj["speaker_id"]
            if spk not in speaker_objs:
                speaker_objs[spk] = list()
            speaker_objs[spk].append(obj)
        
        for spk_id, spk_objs in speaker_objs.items():
            split_spk_utts(
                spk_objs,
                splits["train"]["csv_writer"],
                splits["val"]["csv_writer"],
                hparams["val_ratio"],
                int(hparams["sentence_len"]),
                spk_id,
            )

        for split in splits:
            splits[split]["writer"].close()

        # train the model
        train_data = prepare_dataset(hparams, splits["train"]["file"])
        val_data = prepare_dataset(hparams, splits["val"]["file"])
        speaker_brain = SpeakerBrain(
            modules=hparams["modules"],
            opt_class=hparams["opt_class"],
            hparams=hparams,
            run_opts={"device": self.device},
            checkpointer=hparams["checkpointer"],
        )

        speaker_brain.epoch_losses = {"TRAIN": [], "VALID": []}
        val_kwargs = hparams["dataloader_options"].copy()
        val_kwargs["shuffle"] = False

        speaker_brain.fit(
            speaker_brain.hparams.epoch_counter,
            train_data,
            val_data,
            train_loader_kwargs=hparams["dataloader_options"],
            valid_loader_kwargs=val_kwargs,
            progressbar=True,
        )

        # save the embedding model and load it
        emb_model_state_dict = speaker_brain.modules.embedding_model.state_dict()
        self.model.mods.embedding_model.load_state_dict(emb_model_state_dict)


def split_spk_utts(
    speaker_objs: list[dict],
    train_writer: csv.writer,
    val_writer: csv.writer,
    ratio: float,
    sentence_len: int,
    spk_id: int,
) -> None:
    """
    Split the given list of speaker utterances into training and validation sets.

    Args:
        speaker_objs: List of speaker utterances from the datafile.
        train_writer: CSV writer for the training datafile.
        val_writer: CSV writer for the validation datafile.
        ratio: Ratio of validation utterances.
        sentence_len: Utterances are split into samples of this length.
        spk_id: Speaker ID, as stored in self.speakers.
    """
    indices = list(range(len(speaker_objs)))
    random_indices = random.sample(indices, len(indices))
    n_val = int(len(speaker_objs) * ratio)

    for idx, random_idx in enumerate(random_indices):
        obj = speaker_objs[random_idx]
        writer = val_writer if idx < n_val else train_writer
        fname = os.path.splitext(os.path.basename(obj["path"]))[0]
        for start in range(0, int(obj["duration"]), sentence_len):
            stop = min(start + sentence_len, obj["duration"])
            writer.writerow(
                [
                    f"{fname}_{start}_{stop}",
                    obj["path"],
                    obj["duration"],
                    float(start),
                    float(stop),
                    obj["label"],
                    spk_id,
                ]
            )

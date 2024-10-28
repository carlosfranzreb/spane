#!/usr/bin/python3
"""Recipe for training speaker embeddings (e.g, xvectors) using the VoxCeleb Dataset.
We employ an encoder followed by a speaker classifier.

To run this recipe, use the following command:
> python train_speaker_embeddings.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:
    hyperparams/train_x_vectors.yaml (for standard xvectors)
    hyperparams/train_ecapa_tdnn.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
"""


import torch
import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch: PaddedBatch, stage: sb.Stage):
        """Computation pipeline based on a encoder + speaker classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        embeddings = self.modules.embedding_model(feats)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions, lens = predictions
        predictions = predictions.to(self.device)
        lens = lens.to(self.device)
        spkid = (
            torch.tensor([int(spkid) for spkid in batch.spk_id_encoded])
            .unsqueeze(1)
            .to(self.device)
        )

        loss = self.hparams.compute_cost(predictions, spkid, lens)
        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        for module in [
            self.modules.compute_features,
            self.modules.mean_var_norm,
            self.modules.embedding_model,
            self.modules.classifier,
        ]:
            for p in module.parameters():
                p.requires_grad = True

        if stage == sb.Stage.VALID:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")
            if epoch > 0:
                old_lr, new_lr = self.hparams.lr_annealing(epoch)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                train_stats = self.train_stats
            else:
                old_lr = 0.0
                train_stats = None
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def prepare_dataset(hparams: dict, datafile: str) -> DynamicItemDataset:
    "Creates the datasets and their data processing pipelines."

    data = DynamicItemDataset.from_csv(csv_path=datafile)

    # define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start", "stop")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav: torch.Tensor, start: str, stop: str) -> torch.Tensor:
        start, stop = float(start), float(stop)
        start_frame = int(start * hparams["sample_rate"])
        num_frames = int((stop - start) * hparams["sample_rate"])
        sig, _ = torchaudio.load(wav, num_frames=num_frames, frame_offset=start_frame)
        sig = sig.transpose(0, 1).squeeze(1)
        return sig

    sb.dataio.dataset.add_dynamic_item([data], audio_pipeline)
    sb.dataio.dataset.set_output_keys([data], ["id", "sig", "spk_id_encoded"])

    return data

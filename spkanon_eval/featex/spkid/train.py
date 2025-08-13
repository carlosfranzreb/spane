"""
Recipe for training speaker embeddings based on Speechbrain's.

It includes a target classifier that informs us about how much target information
is encoded by the speaker recognizer. The recognizer can be encouraged to remove
target information with adversarial learning.
"""

import torch
from torch import nn, Tensor
from torch.autograd import Function
import torchaudio
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.batch import PaddedBatch


class GradReverse(Function):
    """
    Inspired by
    https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3

    It should be called in the forward pass of your nn.Module like this:
        x = GradReverse.apply(x, weight)

    x is the tensor with the data; weight is the weighing factor to be applied only
    in the backward pass. It is often called lambda in the literature.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: float) -> Tensor:
        """
        Store the passed weight in the context, to be used in the backward pass.
        """
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        """
        Return two values, one per argument of the forward pass.
        """
        return grad_output.neg() * ctx.weight, None


class AdversarialClassifier(nn.Module):
    def __init__(self, classifier: nn.Module, weight: float):
        super().__init__()
        self.classifier = classifier
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        x = GradReverse.apply(x, self.weight)
        return self.classifier(x)


class SpeakerBrain(sb.core.Brain):
    """Class for speaker embedding training"""

    def compute_forward(self, batch: PaddedBatch, stage: sb.Stage):
        """Computation pipeline based on a encoder + speaker classifier."""
        batch = batch.to(self.device)
        wavs, lens = batch.sig
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)
        embeddings = self.modules.embedding_model(feats)
        outputs_src = self.modules.classifier_src(embeddings)
        outputs_tgt = self.modules.classifier_tgt(embeddings)
        return outputs_src, outputs_tgt, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label."""
        predictions_src, predictions_tgt, lens = predictions
        predictions_src = predictions_src.to(self.device)
        predictions_tgt = predictions_tgt.to(self.device)
        lens = lens.to(self.device)

        sources = (
            torch.tensor([int(spk) for spk in batch.source_speaker])
            .unsqueeze(1)
            .to(self.device)
        )
        targets = (
            torch.tensor([int(spk) for spk in batch.target_speaker])
            .unsqueeze(1)
            .to(self.device)
        )

        loss_src = self.hparams.compute_cost(predictions_src, sources, lens)
        loss_tgt = self.hparams.compute_cost(predictions_tgt, targets, lens)
        loss = (loss_src + loss_tgt) / 2

        if stage == sb.Stage.TRAIN and hasattr(
            self.hparams.lr_annealing, "on_batch_end"
        ):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics_src.append(batch.id, predictions_src, sources, lens)
            self.error_metrics_tgt.append(batch.id, predictions_tgt, targets, lens)

        return loss

    def on_stage_start(self, stage: sb.Stage, epoch: int):
        """Gets called at the beginning of an epoch."""
        if stage == sb.Stage.VALID:
            self.error_metrics_src = self.hparams.error_stats()
            self.error_metrics_tgt = self.hparams.error_stats()
            grad = False
        else:
            grad = True

        # pre-pend GRL to the target classifier and update the weight
        if stage.value == 1 and epoch == 1:
            self.modules.classifier_tgt = AdversarialClassifier(
                self.modules.classifier_tgt, 0.0
            )
        self.modules.classifier_tgt.weight = self.al_weights[epoch - 1]

        for module in [
            self.modules.compute_features,
            self.modules.mean_var_norm,
            self.modules.embedding_model,
            self.modules.classifier_src,
            self.modules.classifier_tgt,
        ]:
            for p in module.parameters():
                p.requires_grad = grad

    def on_stage_end(self, stage: sb.Stage, stage_loss, epoch: int):
        """Gets called at the end of an epoch."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif stage == sb.Stage.VALID:
            stage_stats["error_rate_src"] = self.error_metrics_src.summarize("average")
            stage_stats["error_rate_tgt"] = self.error_metrics_tgt.summarize("average")
            if epoch > 0:
                old_lr, new_lr = self.hparams.lr_annealing(epoch)
                sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
                train_stats = self.train_stats
            else:
                old_lr = 0.0
                train_stats = None
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr": old_lr,
                    "tgt_weight": self.modules.classifier_tgt.weight,
                },
                train_stats=train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"error_rate_src": stage_stats["error_rate_src"]},
                min_keys=["error_rate_src"],
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
    sb.dataio.dataset.set_output_keys(
        [data], ["id", "sig", "source_speaker", "target_speaker"]
    )

    return data

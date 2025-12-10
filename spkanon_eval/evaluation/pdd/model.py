import os

import torch
from torch import Tensor
import torchaudio

from .multihead_attention import MHSA


class PdDetector(torch.nn.Module):
    """
    Inference model for PD detection. Given an audiofile, it predicts PD with the
    model that was trained without the given file.

    `ckpt_dir` should contain the checkpoints for all folds, e.g. `ckpt_fold_0.pth`,
    and their corresponding statistics (mean and std) from wav2vec features the model
    was trained with, stored as a dict, e.g. `train_stats_fold_0.pt`
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str,
        w2v_layer: int = 7,
        latent_dim: int = 1024,
        n_heads: int = 1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.w2v_layer = w2v_layer
        self.w2v = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model().to(device)

        self.stats = list()
        self.mha = list()
        self.classifier = list()
        for fold_idx in range(5):
            fold_stats_f = os.path.join(ckpt_dir, f"train_stats_fold_{fold_idx}.pt")
            fold_stats = torch.load(fold_stats_f, weights_only=False)
            fold_stats = {k: torch.tensor(v).to(device) for k, v in fold_stats.items()}
            self.stats.append(fold_stats)

            fold_mha = MHSA(latent_dim=latent_dim, num_heads=n_heads)
            fold_classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(latent_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(latent_dim, n_classes, bias=False),
            )

            ckpt_f = os.path.join(ckpt_dir, f"ckpt_fold_{fold_idx}.pth")
            ckpt = torch.load(ckpt_f, map_location=device)
            fold_mha.load_state_dict(
                {k.partition("mha.")[2]: v for k, v in ckpt.items() if "mha" in k}
            )
            fold_classifier.load_state_dict(
                {
                    k.partition("classifier.")[2]: v
                    for k, v in ckpt.items()
                    if "classifier" in k
                }
            )
            self.mha.append(fold_mha)
            self.classifier.append(fold_classifier)

    def forward(self, x: Tensor, lens: Tensor, folds: Tensor) -> Tensor:
        """
        Args:
            `x`: waveforms, shape (bs, max_samples)
            `lens`: lengths of waveforms, shape (bs,)
            `folds`: fold of each waveform, shape (bs,)

        Returns:
            `x`: PD predictions, shape (bs, 2). For each waveform, the first value of
                prediction corresponds to the healthy class, and the second to the
                PD class.
        """

        # extract features and compute padding mask
        x, w2v_lens = self.w2v.extract_features(x, lens)
        x = x[self.w2v_layer]
        mask = make_pad_mask(w2v_lens)

        # make the predictions for each fold
        out = torch.empty((x.shape[0], 2))
        for fold in torch.unique(folds):
            filter_for_fold = folds == fold
            x_fold = x[filter_for_fold]
            mask_fold = mask[filter_for_fold]

            x_fold = (x_fold - self.stats[fold]["median"]) / self.stats[fold]["std"]
            x_fold = self.mha[fold](x_fold, x_fold, x_fold, mask_fold)
            x_fold = x_fold.sum(dim=1)
            x_fold = x_fold / w2v_lens[filter_for_fold].unsqueeze(1)
            x_fold = self.classifier[fold](x_fold).squeeze(1)

            out[filter_for_fold] = x_fold

        return out


def make_pad_mask(lengths: Tensor) -> Tensor:
    """
    Input lengths has shape (batch_size)
    Output mask has shape (batch_size, 1, max_features)
    """
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.unsqueeze(1)

    return mask

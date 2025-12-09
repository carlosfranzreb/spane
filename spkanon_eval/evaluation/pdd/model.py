import torch
from torch import Tensor
import torchaudio

from .multiheaded_attention import MHSA


class PdDetector(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        stats_f: str,
        w2v_layer: int = 7,
        latent_dim: int = 1024,
        n_heads: int = 1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.w2v_layer = w2v_layer
        self.w2v = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()

        self.stats = torch.load(stats_f, weights_only=False)
        self.stats = {k: torch.tensor(v) for k, v in self.stats.items()}

        self.mha = MHSA(latent_dim=latent_dim, num_heads=n_heads)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(latent_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(latent_dim, n_classes, bias=False),
        )

        ckpt = torch.load(ckpt, map_location="cpu")
        self.mha.load_state_dict(
            {k.partition("mha.")[2]: v for k, v in ckpt.items() if "mha" in k}
        )
        self.classifier.load_state_dict(
            {
                k.partition("classifier.")[2]: v
                for k, v in ckpt.items()
                if "classifier" in k
            }
        )

    def forward(self, x: Tensor, lens: Tensor) -> Tensor:

        x, w2v_lens = self.w2v.extract_features(x, lens)
        x = x[self.w2v_layer]
        x = (x - self.stats["median"]) / self.stats["std"]
        mask = make_pad_mask(w2v_lens)

        x = self.mha(x, x, x, mask)
        x = x.sum(dim=1)
        x = x / w2v_lens.unsqueeze(1)
        x = self.classifier(x).squeeze(1)

        return x


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

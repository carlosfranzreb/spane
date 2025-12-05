import torch
from torch import Tensor
import torchaudio

from .multiheaded_attention import MultiHeadedAttention


class PdDetector(torch.nn.Module):
    def __init__(
        self,
        ckpt: str,
        w2v_layer: int = 7,
        latent_dim: int = 1024,
        n_heads: int = 1,
        n_classes: int = 2,
    ):
        super().__init__()

        self.w2v_layer = w2v_layer
        self.w2v = torchaudio.pipelines.WAV2VEC2_XLSR_300M.get_model()

        self.mha = MultiHeadedAttention(
            query_dim=latent_dim,
            key_dim=latent_dim,
            value_dim=latent_dim,
            num_heads=n_heads,
            attn_type="self",
        )
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
        x = self.w2v.extract_features(x)[0][self.w2v_layer]
        
        w2v_lens = lens // 320
        w2v_lens = w2v_lens.to(torch.long)
        mask = make_pad_mask(w2v_lens)
        if mask.shape[1] == x.shape[1] + 1:
            mask = mask[:, :-1]

        x = self.mha(x, x, x, mask)
        x = x.mean(dim=1).unsqueeze(1)
        x = self.classifier(x).squeeze(1)
        return x


def make_pad_mask(lengths: Tensor) -> Tensor:
    bs = lengths.shape[0]
    maxlen = lengths.max()

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    return mask

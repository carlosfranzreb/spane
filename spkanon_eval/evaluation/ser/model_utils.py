"""Model classes used by the SER model."""

import torch
from torch import Tensor, nn
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from spkanon_eval.utils import make_pad_mask


class RegressionHead(nn.Module):
    """Classification head."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    """Speech emotion classifier."""

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self, input_values: Tensor, attn_mask: Tensor, audio_lens: Tensor
    ) -> tuple[Tensor, Tensor]:
        outputs = self.wav2vec2(input_values, attn_mask)
        hidden_states = outputs[0]

        # get the feature lens
        feat_lens = self.wav2vec2._get_feat_extract_output_lengths(audio_lens)
        feat_mask = make_pad_mask(feat_lens).squeeze(1)

        hidden_states[feat_mask] = float("nan")
        hidden_states = torch.nanmean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits

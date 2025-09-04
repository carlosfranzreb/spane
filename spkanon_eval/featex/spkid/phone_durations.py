"""
Feature extractor that computes phone durations, based on:

Tomashenko, Natalia, et al. “Exploiting Context-Dependent Duration Features for Voice
Anonymization Attack Systems.” Interspeech 2025.

This feature extractor can be used to replace the spectrogram when training the speaker
recognizer, to assess how much speaker identity is being leaked through phone durations
alone, without any additional speaker information.

! It requires installing private kNN-VC through the spkanon_models repository:
<https://github.com/carlosfranzreb/spkanon_models>

To use it, the spkid model should be trained with config
`spane/config/components/asv/spkid/train_ecapa_phone_durations.yaml`
"""

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from speechbrain.utils.autocast import fwd_default_precision
from omegaconf import OmegaConf

from spkanon_eval.setup_module import setup as setup_module
from spkanon_models.knnvc_private.conv_decoder import load_model as load_conv_decoder


class PhoneDurations(torch.nn.Module):
    """
    Extract phone durations from utterances, with the phone recognizer from private
    kNN-VC.

    Args:
        device: cpu or cuda
        phone_lexicon: text file with one phone per line. It should be the same lexicon
            the predictor was trained on.
        wavlm_ckpt: checkpoint for wavlm
        phone_predictor_ckpt: checkpoint for private kNN-VC's phone decoder.
    """

    def __init__(
        self,
        device: str,
        phone_lexicon: str,
        wavlm_ckpt: str,
        phone_predictor_ckpt: str,
    ):
        super().__init__()
        self.device = device
        self.phone_lexicon = open(phone_lexicon, "r").read().splitlines()
        self.phone_predictor = load_conv_decoder(phone_predictor_ckpt, device)

        wavlm_cfg = OmegaConf.create(
            {
                "cls": "spkanon_eval.featex.wavlm.wrapper.WavlmWrapper",
                "ckpt": wavlm_ckpt,
                "layer": 24,
                "hop_length": 320,
            }
        )
        self.wavlm = setup_module(wavlm_cfg, device)

    @torch.inference_mode()
    @fwd_default_precision(cast_inputs=torch.float32)
    def forward(self, wav: Tensor) -> Tensor:
        """
        Returns a set of features generated from the input waveforms.

        Args:
            wav: batch of audio signals to transform to features.

        Returns:
            feats: sequence of one-hot vectors of the size of the number of phones.
                Instead of a 1, the number in the vector is the phone's duration.
        """
        batch = [wav, None, torch.ones(wav.shape[0], dtype=torch.int) * wav.shape[1]]
        feats, feat_lengths = self.wavlm.run(batch).values()
        phones = self.phone_predictor(feats).argmax(dim=2)

        # compute the phone durations
        feats = list()
        for utt_idx in range(phones.shape[0]):
            unique_phones, phone_durations = torch.unique_consecutive(
                phones[utt_idx], return_counts=True
            )
            utt_feats = torch.zeros(
                (unique_phones.shape[0], len(self.phone_lexicon)),
                dtype=torch.long,
                device=self.device,
            )
            utt_feats[torch.arange(utt_feats.shape[0]), unique_phones] = phone_durations
            feats.append(utt_feats)

        return pad_sequence(feats, batch_first=True).to(torch.float)

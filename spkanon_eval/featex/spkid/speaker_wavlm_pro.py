"""
 * Software Name : spk_embeddings.py
 * SPDX-FileCopyrightText: Copyright (c) Orange SA
 * SPDX-License-Identifier: CC-BY-SA-3.0
 *
 * This software is distributed under the Creative Commons Attribution Share Alike 3.0 Unported,
 * see the "LICENSE.txt" file for more details or https://huggingface.co/Orange/Speaker-wavLM-pro/blob/main/LICENSE.txt
"""

import torch, torchaudio
import torch.nn as nn
from transformers.models.wavlm.modeling_wavlm import WavLMPreTrainedModel, WavLMModel

class TopLayers(nn.Module):
    def __init__(self, embd_size = 250, top_interm_size = 512):
        super(TopLayers, self).__init__()
        self.affine1 = nn.Conv1d(in_channels=2048, out_channels=top_interm_size, kernel_size=1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=top_interm_size, affine=False, eps=1e-03)
        self.affine2 = nn.Conv1d(in_channels=top_interm_size, out_channels=embd_size, kernel_size=1)
        self.batchnorm2 = nn.BatchNorm1d(num_features=embd_size, affine=False, eps=1e-03)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.batchnorm1(self.activation(self.affine1(x)))
        out = self.batchnorm2(self.activation(self.affine2(out)))
        return nn.functional.normalize(out[:,:,0])
    
class EmbeddingsModel(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wavlm = WavLMModel(config)
        self.top_layers = TopLayers(config.embd_size, config.top_interm_size)
        
    def forward(self, input_values):
        # MVN normalization
        x_norm = (input_values - input_values.mean(dim=1).unsqueeze(1)) / (input_values.std(dim=1).unsqueeze(1))
        # wavlm fwd
        base_out = self.wavlm(input_values=x_norm, output_hidden_states=False).last_hidden_state
        # stats pooling
        v = base_out.var(dim=1).clamp(min=1e-10)
        x_stats = torch.cat((base_out.mean(dim=1),v.pow(0.5)),dim=1).unsqueeze(dim=2)
        # top layers fwd 
        return self.top_layers(x_stats)
    

def compute_embedding(fnm, model, max_size=320000):
    sig, sr = torchaudio.load(fnm)
    assert sr == 16000, "please convert your audio file to a sampling rate of 16 kHz"
    sig = sig.mean(dim=0)
    if sig.shape[0] > max_size:
        print(f"truncating long signal {fnm}")
        sig = sig[:max_size]
    embd = model(sig.unsqueeze(dim=0))
    return embd.clone().detach()
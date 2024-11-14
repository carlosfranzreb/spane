import torch
from omegaconf import OmegaConf, ListConfig

from .wavlm_model import WavLM, WavLMConfig

from spkanon_eval.component_definitions import InferComponent


class WavlmWrapper(InferComponent):
    def __init__(self, config: OmegaConf, device: str, **kwargs):
        """
        Load the model from the checkpoint and move it to the device.

        Args:
            config: the configuration object. should include the path to the ckpt
            as `ckpt_path` and the layer from which to extract the features as
            `layer`.
            device: the device where the model should run.
        """
        ckpt = torch.load(config.ckpt, map_location=device)
        model_cfg = WavLMConfig(ckpt["cfg"])
        self.model = WavLM(model_cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.config = config

    @torch.inference_mode()
    def run(self, batch: list) -> dict:
        """
        Pases the batch through the model until the specified layer and returns the
        output of that layer.

        Args:
            batch: a list with a tensor comprising waveforms in first position, and the
            number of samples per item in the batch in third position.

        Returns:
            a dictionary with the output of the specified WavLM layer under the key "feats" and
            the number of feats for each sample under "n_feats".
        """
        with torch.no_grad():
            if isinstance(self.config.layer, ListConfig):
                last_feats, all_feats = self.model.extract_features(
                    batch[0],
                    ret_layer_results=True,
                    output_layer=max(self.config.layer),
                )[0]
                out = list()
                for l in self.config.layer:
                    out.append(all_feats[l][0].transpose(0, 1))
                out = torch.stack(out)
            else:
                out = self.model.extract_features(
                    batch[0], output_layer=self.config.layer
                )[0]

        n_feats = batch[2] // self.config.hop_length
        return {"feats": out, "n_feats": n_feats}

    def to(self, device: str) -> None:
        """
        Implementation of PyTorch's `to()` method to set the device.
        """
        self.device = device
        self.model.to(self.device)

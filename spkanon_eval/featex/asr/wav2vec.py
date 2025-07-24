from speechbrain.lobes.models.huggingface_transformers.wav2vec2 import Wav2Vec2
from omegaconf import DictConfig
import torch
from torch import Tensor
from spkanon_eval.component_definitions import InferComponent


class Wav2Vec(InferComponent):
    def __init__(self, config: DictConfig, device: str, **kwargs) -> None:
        self.model = Wav2Vec2(
            config["hf_hub"],
            config["save_path"],
        )

    @torch.inference_mode()
    def run(self, batch: list) -> Tensor:
        return self.model(batch[0])

from abc import ABC, abstractmethod

from omegaconf import DictConfig

from spkanon_eval.datamodules import AudioBatch


class Component(ABC):
    @abstractmethod
    def __init__(self, config: DictConfig, device: str, **kwargs) -> None:
        pass


class InferComponent(Component):
    @abstractmethod
    def run(self, batch: AudioBatch | dict) -> dict:
        pass

    @abstractmethod
    def to(self, device: str) -> None:
        pass


class EvalComponent(Component):
    @abstractmethod
    def train(self, exp_folder: str) -> None:
        pass

    @abstractmethod
    def eval_dir(self, exp_folder: str, datafile: str, is_baseline: bool) -> None:
        pass

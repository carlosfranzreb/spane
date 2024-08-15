import logging

import torch
from torch import Tensor
from omegaconf import DictConfig

from spkanon_eval.setup_module import setup


LOGGER = logging.getLogger("progress")


class Anonymizer:
    def __init__(self, config: DictConfig, log_dir: str) -> None:
        """Initialize the components and optionally the target selection algorithm."""
        super().__init__()
        self.config = config
        self.device = config.device
        self.log_dir = log_dir

        for module in ["featex", "featproc", "featfusion", "synthesis"]:
            cfg = config.get(module, None)
            if cfg is not None:
                setattr(self, module, setup(cfg, self.device))
            else:
                setattr(self, module, None)

        # if possible, get the output of the featproc module
        if self.featproc is not None:
            self.proc_out = self.featproc.pop("output")

        # if there is a target selection algorithm, pass it to the right component
        target_selection_cfg = config.get("target_selection", None)
        if target_selection_cfg is not None and self.featproc is not None:
            args = [target_selection_cfg]
            if hasattr(target_selection_cfg, "extra_args"):
                for arg in target_selection_cfg.extra_args:
                    args.append(getattr(self, arg))
            for name, component in self.featproc.items():
                if hasattr(component, "target_selection"):
                    LOGGER.info(f"Passing target selection algorithm to {name}")
                    component.init_target_selection(*args)

    def get_feats(self, batch: list, source: Tensor, source_is_male: Tensor) -> dict:
        """
        Run the featex, featproc and featfusion modules. Returns anonymized features and
        targets. Sources refer to the input speaker, and targets to the output speaker.
        """
        out = batch
        if self.featex is not None:
            out = self._run_module(self.featex, batch)

        out["source"] = source
        out["source_is_male"] = source_is_male
        if self.featproc is not None:
            processed = self._run_module(self.featproc, out)
            out_proc = dict()
            for feat in self.proc_out["featex"]:
                out_proc[feat] = out[feat]
            for feat in self.proc_out["featproc"]:
                out_proc[feat] = processed[feat]
            out = out_proc
        if self.featfusion is not None:
            out = self.featfusion.run(out)
        return out

    def forward(self, batch: list, data: list) -> tuple[Tensor, Tensor, Tensor]:
        """Returns anonymized speech, item lengths and targets."""
        source = torch.tensor(
            [d["speaker_id"] for d in data], dtype=torch.long, device=self.device
        )
        if "gender" in data[0]:
            source_is_male = torch.tensor(
                [d["gender"] == "M" for d in data], dtype=torch.bool, device=self.device
            )
        else:
            LOGGER.warning(
                """
                gender information not found in the data. Make sure it is not
                considered for target selection.
                """
            )
            source_is_male = torch.zeros_like(source, dtype=torch.bool)

        with torch.no_grad():
            out = self.get_feats(batch, source, source_is_male)
            waves, n_samples = self.synthesis.run(out)
        return waves, n_samples, out["target"]

    def _run_module(self, module: dict, batch: list) -> dict:
        """
        Run each component of the module with the given batch. The outputs are stored
        in a dictionary. If the output of a component is a dictionary, its keys are
        used as keys in the output dictionary. Otherwise, the component name is used
        as key.
        """
        out = dict()
        for name, component in module.items():
            component_out = component.run(batch)
            if isinstance(component_out, dict):
                out.update(component_out)
            else:
                out[name] = component_out
        return out

    def set_consistent_targets(self, consistent_targets: bool) -> None:
        """
        Set whether the selected targets should be consistent. We assume that targets
        are selected in the `featproc` module, and iterate over its components looking
        for the equivalent method.
        """
        if self.featproc is None:
            LOGGER.warning("No featproc module found, so no targets to be consistent")
            return
        for name, component in self.featproc.items():
            if hasattr(component, "target_selection"):
                LOGGER.info(f"Set consistent targets of {name} to {consistent_targets}")
                component.target_selection.set_consistent_targets(consistent_targets)

    def to(self, device: str) -> None:
        """
        Overwrite of PyTorch's method, to propagate it to all components.
        """
        LOGGER.info(f"Changing model's device to {device}")
        for attr in [self.featex, self.featproc, self.featfusion]:
            if attr is not None:
                for component in attr.values():
                    component.to(device)
        self.synthesis.to(device)
        self.device = device

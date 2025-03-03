import os
import logging
import copy

from omegaconf import OmegaConf

from spkanon_eval.setup_module import setup
from spkanon_eval.utils import seed_everything
from spkanon_eval.anonymizer import Anonymizer


SAMPLE_RATE = 16000  # sample rate for evaluation
LOGGER = logging.getLogger("progress")


def evaluate(exp_folder: str, model: Anonymizer, config: OmegaConf) -> None:
    """
    Evaluate the given experiment with the components defined in the config. Components
    may be trained before the evaluation.

    Args:
        exp_folder: path to the experiment folder
        model: the anonymizer model
        config: the config object, as defined in the documentation (TODO)
    """

    # change the RNG seed if required
    if "seed" in config.eval.config:
        seed_everything(config.eval.config.seed)

    # ensure that the directory where the evaluation results will be stored exists
    os.makedirs(os.path.join(exp_folder, "eval"), exist_ok=True)

    # use the anonymized datafiles if we are not evaluating the baseline
    is_baseline = config.eval.config.baseline
    fname = "eval" if is_baseline is True else "anon_eval"
    datafile = os.path.join(exp_folder, "data", f"{fname}.txt")

    # iterate over the components and train & evaluate them
    for name, cfg in config.eval.components.items():
        cfg.data = copy.deepcopy(config.data)
        component = setup(cfg, config.device, model=model)

        if cfg.train is True:
            LOGGER.info(f"Training component `{name}`")
            component.train(exp_folder)

        LOGGER.info(f"Running evaluation with component `{name}`")
        component.eval_dir(exp_folder, datafile, is_baseline)

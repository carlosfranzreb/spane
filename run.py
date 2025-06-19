from argparse import ArgumentParser
import os
import logging
import yaml
from time import time, sleep
import subprocess

from omegaconf import OmegaConf, DictConfig

from spkanon_eval.main import main
from spkanon_eval.utils import seed_everything


def setup(args) -> DictConfig:
    config = load_subconfigs(yaml.full_load(open(args.config)))
    config = OmegaConf.create(config)
    config.device = args.device
    config.data.config.num_workers = args.num_workers
    config.commit_hash = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    override_with_args(config, args.config_overrides)

    # create the experiment folder (a.k.a. logging directory)
    exp_folder = os.path.join(config.log_dir, str(int(time())))
    while os.path.exists(exp_folder):
        sleep(1)
        exp_folder = os.path.join(config.log_dir, str(int(time())))
    os.makedirs(exp_folder)
    config.exp_folder = exp_folder

    # if a seed is specified, set it
    if config.seed is not None:
        seed_everything(config.seed)

    # dump config file to experiment folder
    OmegaConf.save(config, os.path.join(exp_folder, "exp_config.yaml"))

    # create logger in experiment folder to log progress: dump to file and stdout
    logger_name = "progress"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(exp_folder, f"{logger_name}.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    logger.addHandler(logging.StreamHandler())

    return config


def load_subconfigs(config: DictConfig) -> DictConfig:
    """
    Given a config, load all the subconfigs that are specified in the config into
    the same level as the parameter. Configs are specified by parameters ending with
    '_cfg'. If a value of the config is a dict, call this function again recursively.
    """

    full_config = dict()
    for key, value in config.items():
        if isinstance(value, dict):
            full_config[key] = load_subconfigs(value)
        elif key.endswith("_cfg"):
            full_config.update(load_subconfigs(yaml.full_load(open(value))))
        else:
            full_config[key] = value

    return full_config


def override_with_args(config: DictConfig, args: dict):
    """
    Update the config with the passed arguments. The new value is casted to the type of
    the existing value. If that fails, an error is raised.
    """

    def check_key_existence(config: DictConfig, key: str, full_key: str):
        """Raise an error if the key does not exist in the cofig."""
        if key not in config:
            raise KeyError(
                f"Subkey '{key}' not found in config for override '{full_key}'"
            )

    for key, value in args.items():
        keys = key.split(".")

        # iterate over the keys that serve as directories
        sub_config = config
        for sub_key in keys[:-1]:
            check_key_existence(sub_config, sub_key, key)
            sub_config = sub_config[sub_key]

        # change the value of the last key
        last_key = keys[-1]
        check_key_existence(sub_config, last_key, key)
        if type(sub_config[last_key]) is not type(value):
            old_type = type(sub_config[last_key])
            try:
                value = old_type(value)
            except Exception as e:
                raise ValueError(
                    f"Failed to cast override for '{key}' to {old_type}: {e}"
                )
        sub_config[last_key] = value


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=10, type=int)
    args, config_overrides = parser.parse_known_args()
    if len(config_overrides) % 2 != 0:
        raise RuntimeError(
            "The number of config arguments must be even (key-value pairs)"
        )

    # add the config arguments to the args
    args.config_overrides = dict()
    for key_idx in range(0, len(config_overrides), 2):
        args.config_overrides[config_overrides[key_idx]] = config_overrides[key_idx + 1]

    main(setup(args))

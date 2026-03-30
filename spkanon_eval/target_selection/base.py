import logging
from typing import Callable
import json

import torch
from torch import Tensor
from omegaconf import DictConfig

LOGGER = logging.getLogger("progress")


class BaseSelector:
    def __init__(self, cfg: DictConfig, target_df: str = None) -> None:
        """
        Initialize the target selector with configuration parameters and target metadata.

        Args:
            cfg (DictConfig): Configuration dictionary containing selection parameters:
                - consistent_targets (bool): If True, stores assignments in a dictionary
                  to ensure targets remain consistent across utterances for the same speaker.
                - same_source_target (bool, optional): If True, checks that source and
                  target IDs differ to ensure anonymization when source and target
                  datasets are identical. Ignored by `FixedSelector`. Defaults to False.
                - target_constraints (dict, optional): Hard filters for target speakers
                  (e.g., {"is_male": True}).
                - conversion_constraints (dict, optional): Constraints relative to the
                  source speaker's attributes. Valid values for each key are:
                  - "same": Target must have the same attribute value as the source.
                  - "opposite": Target must have a different attribute value.
                  - None: No constraint for this attribute.
            target_df (str, optional): Path to the target dataset file (JSONL format)
                containing speaker metadata used to evaluate constraints.

        Raises:
            ValueError: If an invalid value is provided in `conversion_constraints`.
        """
        self.cfg = cfg
        self.targets = dict() if cfg.consistent_targets else None
        self.same_source_target = cfg.get("same_source_target", False)
        self.target_info = {"speaker_id": get_spk_attr(target_df, "speaker_id")}

        # keep only the targets that fulfill the target constraints
        target_constraints = cfg.get("target_constraints", dict())
        target_mask = torch.ones(
            self.target_info["speaker_id"].shape[0], dtype=torch.bool
        )
        for key, value in target_constraints.items():
            mask_func = lambda spk_value: spk_value == value
            target_mask &= get_spk_attr(target_df, key, mask_func)

        self.target_info["speaker_id"] = self.target_info["speaker_id"][target_mask]

        # check conversion constraints and store required metadata
        self.conversion_constraints = cfg.get("conversion_constraints", dict())
        for key, value in self.conversion_constraints.items():
            # check that the value is correct
            if value not in [None, "same", "opposite"]:
                error = f"Invalid value for the constraint `{key}`"
                LOGGER.error(error)
                raise ValueError(error)

            # if needed, get the target speaker's values for the key
            if value is not None:
                self.target_info[key] = get_spk_attr(target_df, key)[target_mask]

    def select(self, batch: dict) -> Tensor:
        """
        Select targets for the speakers in the batch. The batch is the same as the one
        output by the dataset (see `spkanon_eval.datamodules.dataset`).

        If speaker consistency is enabled, the target speakers must be consistent
        across the utterances of each speaker.
        """

        n_utts = batch["source"].shape[0]
        device = batch["source"].device
        source = batch["source"].to(device)

        # if speaker consistency is disabled, select new targets and return them
        if self.targets is None:
            mask = torch.ones_like(source, dtype=torch.bool, device="cpu")
            return self.select_new(mask, batch)

        # find the unique source speakers in the batch (TODO: test this on GPU)
        src_mask = torch.zeros_like(source, dtype=torch.bool, device="cpu")
        added_source_spk = list()
        for idx, src in enumerate(source):
            src = src.item()
            if src not in self.targets and src not in added_source_spk:
                src_mask[idx] = True
                added_source_spk.append(src)

        # select new targets for the new unique source speakers
        if torch.sum(src_mask) > 0:
            new_targets = self.select_new(src_mask, batch)

        # create the output targets and store the assignments if needed
        target = torch.ones(n_utts, dtype=torch.int64, device=device)
        for idx, src in enumerate(source):
            src = src.item()
            if src in self.targets:
                target[idx] = self.targets[src]
            else:
                target[idx] = new_targets[added_source_spk.index(src)]
                self.targets[src] = target[idx].item()

        return target

    def select_new(self, indices: Tensor, batch: dict) -> Tensor:
        """
        Select a new target speaker style vector for the batch speakers of the given
        indices. `input_cfg` refers the component's configuration named "input", where
        the input names for the component are defined.
        """
        raise NotImplementedError

    def get_consistent_targets(self) -> bool:
        """Return whether targets are consistent across speakers."""
        return self.targets is not None

    def set_consistent_targets(self, consistent_targets: bool) -> None:
        """
        Update the target selection algorithm with the new value of
        `consistent_targets`.
        """
        if consistent_targets is True:
            LOGGER.info("Enabling consistent targets and removing previous targets")
            self.targets = dict()
        elif consistent_targets is False:
            LOGGER.info("Disabling consistent targets")
            self.targets = None


def get_candidate_target_mask(
    target_values: Tensor, source_values: Tensor, target_flag: str
) -> Tensor:
    """
    If the given flag is enforced, return a mask of the target speakers that are
    eligible for each of the source speakers. Currently, two flags use this function:
    `gender_conversion` and `fold_conversion` (in the PddSelector).

    Args:
        source_values: tensor of shape (batch_size,) stating their values for the flag.
        target_values: tensor of shape (n_targets,) stating their values for the flag.
        target_flag: can have 3 values:
            - "same": source and target values have to be the same.
            - "opposite": source and target values must differ.
            - null: the relationship between source and target values does not matter;
                all targets are eligible.

    Returns:
        boolean tensor of shape (n_targets, batch_size) stating which target speakers
        are eligible for each source speaker.
    """

    if target_flag is None:
        return torch.ones(
            (target_values.shape[0], source_values.shape[0]), dtype=torch.bool
        )
    elif target_flag == "same":
        return torch.eq(target_values.unsqueeze(1), source_values)
    elif target_flag == "opposite":
        return ~torch.eq(target_values.unsqueeze(1), source_values)
    else:
        error = "Invalid value for the `target_flag` parameter"
        LOGGER.error(error)
        raise ValueError(error)


def get_spk_attr(df: str, df_key: str, transform: Callable = None) -> Tensor:
    """
    Get the metadata field `df_key` from each speaker in the datafile `df`. Apply the
    function `transform` to each value before storing it. This function can be used
    to convert strings into numericals or booleans, e.g. gender labels to "is_male"
    flags.
    """
    out = dict()

    # get the transformed attributes for all speakers
    for line in open(df):
        obj = json.loads(line)
        if obj["speaker_id"] in out:
            continue

        value = obj[df_key]
        if transform:
            value = transform(value)

        out[obj["speaker_id"]] = value

    # transform the dict to a tensor, sorted by speaker ID
    out = [out[key] for key in sorted(out.keys())]
    out = torch.tensor(out)

    return out

import logging
import torch
from torch import Tensor
from omegaconf import DictConfig


LOGGER = logging.getLogger("progress")


class BaseSelector:
    def __init__(
        self, vecs: list, cfg: DictConfig, target_is_male: Tensor = None
    ) -> None:
        """
        Initialize the target selector with the style vectors and a flag indicating
        whether the targets should be consistent across the utterances. If this flag
        is set to True, the targets are stored in a dictionary. This class takes care
        of updating the dictionary with the new targets and decides when a new target
        has to be selected. Selectors that inherit this class must implement the
        `select_new` method, where a new target is selected for a given utterance.

        Gender conversion is optional and can be enabled by setting the `gender_conversion`
        parameter. There are two possible values for this parameter:
        - same: the target speaker must have the same gender as the source speaker.
        - opposite: the target speaker must have the opposite gender as the source speaker.
        """
        self.vecs = vecs
        self.targets = dict() if cfg.consistent_targets else None
        self.target_is_male = target_is_male
        self.gender_conversion = cfg.get("gender_conversion", None)
        if self.gender_conversion is not None:
            if self.target_is_male is None:
                error = "Target genders are required by the target selector"
                LOGGER.error(error)
                raise ValueError(error)
            elif self.gender_conversion not in ["same", "opposite"]:
                error = "Invalid value for the gender_conversion parameter"
                LOGGER.error(error)
                raise ValueError(error)

    def select(
        self, source_data: Tensor, source: Tensor, source_is_male: Tensor
    ) -> Tensor:
        """
        If speaker consistency is enabled, the target speakers must be consistent
        across the utterances of each speaker.
        """

        # if speaker consistency is disabled, select new targets and return them
        if self.targets is None:
            return self.select_new(source_data, source_is_male)

        # find the unique source speakers in the batch
        new_sources = list()
        new_source_indices = list()
        for idx, src in enumerate(source):
            src = src.item()
            if src not in self.targets and src not in new_sources:
                new_sources.append(src)
                new_source_indices.append(idx)

        # select new targets for the new unique source speakers
        if len(new_sources) > 0:
            new_targets = self.select_new(
                source_data[new_source_indices], source_is_male[new_source_indices]
            )

        # create the output targets and store the assignments if needed
        target = torch.ones(
            source_data.shape[0], dtype=torch.int64, device=source_data.device
        )
        for idx, src in enumerate(source):
            src = src.item()
            if src in self.targets:
                target[idx] = self.targets[src]
            else:
                target[idx] = new_targets[new_sources.index(src)]
                self.targets[src] = target[idx].item()

        return target

    def select_new(self, source_data: Tensor, source_is_male: Tensor) -> Tensor:
        """
        Select a new target speaker style vector for the given source data.
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

    def get_candidate_target_mask(self, source_is_male: Tensor) -> Tensor:
        """
        If gender conversion is enforced, return a mask of the target speakers
        that are eligible for each of the source speakers.

        Args:
            source_is_male: boolean tensor of shape (batch_size,) stating
                whether the source speakers are male speakers.

        Returns:
            boolean tensor of shape (n_targets, batch_size) stating whether
            the target speakers are eligible for each of the source speakers.
        """
        if self.gender_conversion is None:
            return torch.ones(
                (len(self.vecs), source_is_male.shape[0]), dtype=torch.bool
            )
        elif self.gender_conversion == "same":
            return torch.eq(self.target_is_male.unsqueeze(1), source_is_male)
        elif self.gender_conversion == "opposite":
            return ~torch.eq(self.target_is_male.unsqueeze(1), source_is_male)
        else:
            error = "Invalid value for the gender_conversion parameter"
            LOGGER.error(error)
            raise ValueError(error)

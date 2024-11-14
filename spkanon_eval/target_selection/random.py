import torch
from torch import Tensor

from .base import BaseSelector


class RandomSelector(BaseSelector):
    def select_new(self, source_data: Tensor, source_is_male: Tensor) -> Tensor:
        """
        Randomly select a target for the given input source_data.
        """
        target_mask = self.get_candidate_target_mask(source_is_male)
        targets = torch.zeros(
            source_is_male.shape[0], dtype=torch.int64, device=source_data.device
        )

        for idx in range(source_is_male.shape[0]):
            candidate_indices = target_mask[:, idx].nonzero().flatten()
            sampled_candidate_idx = candidate_indices[
                torch.randint(candidate_indices.shape[0], (1,))
            ]
            targets[idx] = sampled_candidate_idx

        return targets

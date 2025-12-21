import torch
from torch import Tensor

from .base import BaseSelector, get_candidate_target_mask


class RandomSelector(BaseSelector):
    def select_new(self, indices: Tensor, batch: list[Tensor]) -> Tensor:
        """Randomly select a target for the given input source_data."""

        n_utts = indices.shape[0]
        source = batch[self.config.input.source]
        device = batch[self.config.input.feats].device

        # create the target mask from the conversion constraints
        target_mask = torch.ones(n_utts, dtype=torch.bool)
        for key, value in self.cfg.conversion_constraints.items():
            source_info = batch[self.config.input][key].to(device)
            new_target_mask = get_candidate_target_mask(
                self.target_info[key], source_info, value
            ).to(device)
            target_mask &= new_target_mask

        # sample targets
        targets = torch.zeros(n_utts, dtype=torch.int64, device=device)
        for idx, source_spkid in source:
            candidate_indices = target_mask[:, idx].nonzero().flatten()
            if self.same_source_target:
                candidate_indices = candidate_indices[candidate_indices != source_spkid]

            sampled_candidate_idx = candidate_indices[
                torch.randint(candidate_indices.shape[0], (1,))
            ]
            targets[idx] = sampled_candidate_idx

        return targets

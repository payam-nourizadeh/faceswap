"""Source/target pair sampling for training."""

from __future__ import annotations

import random
from typing import Optional


class PairSampler:
    def __init__(self, identity_to_indices: dict[str, list[int]], same_id_prob: float = 0.2, rng: Optional[random.Random] = None,):
        """
        Args:
            identity_to_indices: identity to dataset indices belonging to it
            same_id_prob: probability a step is same-identity (reconstruction).
            rng: random source; defaults to the global `random` (PyTorch seeds
                it per worker) """
        
        if not 0.0 <= same_id_prob <= 1.0:
            raise ValueError("same_id_prob must be in [0, 1]")
        if not identity_to_indices:
            raise ValueError("identity_to_indices must not be empty")

        self.identity_to_indices = identity_to_indices
        self.identities = list(identity_to_indices.keys())
        self.same_id_prob = same_id_prob
        self._rng = rng if rng is not None else random
        self._can_cross = len(self.identities) >= 2

    def sample_source(self,target_identity: str, target_index: int) -> tuple[int, bool]:
        # Return (source_index, is_same_identity) for a target
        want_same = self._rng.random() < self.same_id_prob

        # Falls back to same-identity if we can't cross (only one identity).
        if not want_same and self._can_cross:
            return self._sample_cross(target_identity), False

        return self._sample_same(target_identity, target_index), True

    def _sample_same(self, target_identity: str, target_index: int) -> int:
        members= self.identity_to_indices[target_identity]
        
        if len(members) >= 2:
            # prefer a different image of the same person
            others = [m for m in members if m != target_index]
            return self._rng.choice(others)
        # only one image!! reconstruct from itself
        return members[0]

    def _sample_cross(self, target_identity: str) -> int:
        # Resample until we land on a different identity
        other = target_identity
        while other == target_identity:
            other = self._rng.choice(self.identities)
        return self._rng.choice(self.identity_to_indices[other])
    

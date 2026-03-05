"""
LearningSetting interface: who plays and how reward is derived from scores.

- build_players(learner, env) -> list of players
- learner_index: int
- reward_from_scores(scores_batch, **kwargs) -> reward (batch,)
"""

from typing import Protocol

import torch

from faraway.torch.env import BatchedFarawayEnv, PlayerLike


class LearningSetting(Protocol):
    """Who plays; how reward is derived from scores."""

    learner_index: int

    def build_players(
        self,
        learner: PlayerLike,
        env: BatchedFarawayEnv,
    ) -> list[PlayerLike]:
        """Return list of players for rollout (learner first for solo/vs_random/self_play)."""
        ...

    def reward_from_scores(
        self,
        scores_batch: torch.Tensor,
        *,
        bonus_cards_played: torch.Tensor | None = None,
        main_card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert scores (and optional shaping inputs) to reward tensor (batch,)."""
        ...


class SoloLearningSetting:
    """One player (learner); reward = score + optional shaping (id/bonus/low_id)."""

    learner_index = 0

    def __init__(
        self,
        n_rounds: int,
        use_bonus_cards: bool,
        id_increase_reward_weight: float = 0.0,
        bonus_reward_weight: float = 0.0,
        low_id_reward_weight: float = 0.0,
    ) -> None:
        self.n_rounds = n_rounds
        self.use_bonus_cards = use_bonus_cards
        self.id_increase_reward_weight = id_increase_reward_weight
        self.bonus_reward_weight = bonus_reward_weight
        self.low_id_reward_weight = low_id_reward_weight

    def build_players(
        self,
        learner: PlayerLike,
        env: BatchedFarawayEnv,
    ) -> list[PlayerLike]:
        return [learner]

    def reward_from_scores(
        self,
        scores_batch: torch.Tensor,
        *,
        bonus_cards_played: torch.Tensor | None = None,
        main_card_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reward = scores_batch.float()
        max_increases = self.n_rounds - 1
        mid_increases = max_increases / 2.0

        if self.id_increase_reward_weight > 0 and main_card_ids is not None:
            id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]
            n_increases = id_diffs.float().sum(dim=1)
            increases_normalized = (n_increases - mid_increases) / mid_increases
            reward = reward + self.id_increase_reward_weight * increases_normalized

        if self.bonus_reward_weight > 0 and self.use_bonus_cards and bonus_cards_played is not None:
            max_bonus_cards = self.n_rounds - 1
            mid_bonus_cards = max_bonus_cards / 2.0
            bonus_normalized = (bonus_cards_played - mid_bonus_cards) / mid_bonus_cards
            reward = reward + self.bonus_reward_weight * bonus_normalized

        if self.low_id_reward_weight > 0 and main_card_ids is not None:
            avg_card_id = main_card_ids.mean(dim=1)
            max_card_id = 68.0
            mid_card_id = max_card_id / 2.0
            low_id_normalized = (mid_card_id - avg_card_id) / mid_card_id
            reward = reward + self.low_id_reward_weight * low_id_normalized

        return reward

"""
Rollout: run the environment for one batch and return log-probs, scores, and optional env state.

Used by the trainer to collect (log_probs_learner, scores, env_state) for advantage
and loss computation. Peer-relative advantage needs env_state (n_sub_batches,
current_sub_batch_size).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from faraway.torch.env import BatchedFarawayEnv, PlayerLike, RoundResult


@dataclass
class RolloutResult:
    """Result of one rollout: learner log-probs, scores, and optional env state for advantage."""

    # (batch, n_decisions) log-probability of each decision for the learner
    log_probs_learner: torch.Tensor
    # (batch,) or (batch, n_players) scores; caller typically uses scores[:, learner_index]
    scores: torch.Tensor
    # Optional: for peer-relative advantage (n_sub_batches, current_sub_batch_size)
    env_state: dict[str, int] | None = None


def run_rollout(
    env: BatchedFarawayEnv,
    players: Sequence[PlayerLike],
    learner_index: int,
    temperature: float,
    batch_size: int,
) -> RolloutResult:
    """Run the environment for one batch of games; return log-probs, scores, and optional env state.

    Args:
        env: Batched environment (solo or multiplayer).
        players: List of players; learner is players[learner_index].
        learner_index: Index of the learner in players (0 for solo).
        temperature: Softmax temperature for the learner's policy.
        batch_size: Number of games in the batch.

    Returns:
        RolloutResult with log_probs_learner (batch, n_decisions), scores (batch, n_players),
        and env_state (for peer-relative: n_sub_batches, current_sub_batch_size) or None.
    """
    env.reset(batch_size, players)
    probs_list: list[torch.Tensor] = []

    for _ in range(env.n_rounds):
        result: RoundResult = env.step_round(players, temperature=temperature)
        # Per-round probs for the learner: (batch, n_choices_this_round)
        probs_list.append(result.picked_probabilities[learner_index])

    # (batch, n_decisions)
    picked_probs = torch.cat(probs_list, dim=1)
    log_probs_learner = torch.log(picked_probs.clamp(min=1e-8))

    scores = env.get_scores(players)  # (batch, n_players)

    env_state: dict[str, int] | None = None
    if getattr(env, "peer_relative_reward", False) and env.n_sub_batches > 0:
        env_state = {
            "n_sub_batches": env.n_sub_batches,
            "current_sub_batch_size": env.current_sub_batch_size,
        }

    return RolloutResult(
        log_probs_learner=log_probs_learner,
        scores=scores,
        env_state=env_state,
    )

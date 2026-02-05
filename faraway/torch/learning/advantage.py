"""
Advantage computation: reward -> advantage for policy gradient.

- BaselineEMA: advantage = reward - baseline; baseline updated with EMA.
- PeerRelativeZScore / PeerRelativeCenter: within-sub-batch normalization
  (requires env_state with n_sub_batches, current_sub_batch_size).
"""

import torch


class BaselineEMA:
    """Baseline-based advantage: advantage = reward - baseline; update baseline with EMA."""

    def __init__(
        self,
        prior_baseline: float = 29.0,
        update_rate: float = 0.05,
    ):
        self.baseline = prior_baseline
        self.update_rate = update_rate

    def compute(self, reward: torch.Tensor) -> torch.Tensor:
        """Return advantage = reward - baseline (no in-place update)."""
        return reward - self.baseline

    def update(self, reward: torch.Tensor) -> None:
        """Update baseline with EMA: baseline += update_rate * (reward.mean() - baseline)."""
        new_baseline = self.baseline + self.update_rate * (reward.mean().item() - self.baseline)
        self.baseline = new_baseline


def _peer_relative_advantage(
    reward: torch.Tensor,
    n_sub_batches: int,
    current_sub_batch_size: int,
    normalization: str,  # "zscore" or "center"
) -> torch.Tensor:
    """Compute advantage by normalizing within each sub-batch.

    Games in the same sub-batch faced identical cards. Pad if batch_size
    is not evenly divisible by sub_batch_size.
    """
    batch_size = reward.shape[0]
    padded_size = n_sub_batches * current_sub_batch_size
    if batch_size < padded_size:
        padded_rewards = torch.zeros(padded_size, device=reward.device)
        padded_rewards[:batch_size] = reward
        reshaped = padded_rewards.view(n_sub_batches, current_sub_batch_size)
        valid_mask = torch.zeros(padded_size, dtype=torch.bool, device=reward.device)
        valid_mask[:batch_size] = True
        valid_mask = valid_mask.view(n_sub_batches, current_sub_batch_size)
    else:
        reshaped = reward.view(n_sub_batches, current_sub_batch_size)
        valid_mask = None

    if valid_mask is not None:
        sub_means = (reshaped * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1)
    else:
        sub_means = reshaped.mean(dim=1, keepdim=True)

    if normalization == "zscore":
        centered = reshaped - sub_means
        if valid_mask is not None:
            sub_stds = (
                (centered**2 * valid_mask).sum(dim=1, keepdim=True)
                / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            ).sqrt()
        else:
            sub_stds = reshaped.std(dim=1, keepdim=True)
        normalized = centered / (sub_stds + 1e-8)
    else:
        normalized = reshaped - sub_means

    advantage = normalized.view(-1)[:batch_size]
    return advantage


class PeerRelativeZScore:
    """Peer-relative advantage: z-score within each sub-batch (reward - mean) / std."""

    def compute(
        self,
        reward: torch.Tensor,
        n_sub_batches: int,
        current_sub_batch_size: int,
    ) -> torch.Tensor:
        return _peer_relative_advantage(reward, n_sub_batches, current_sub_batch_size, "zscore")


class PeerRelativeCenter:
    """Peer-relative advantage: center within each sub-batch (reward - mean)."""

    def compute(
        self,
        reward: torch.Tensor,
        n_sub_batches: int,
        current_sub_batch_size: int,
    ) -> torch.Tensor:
        return _peer_relative_advantage(reward, n_sub_batches, current_sub_batch_size, "center")

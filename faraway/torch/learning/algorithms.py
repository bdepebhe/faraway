"""
Algorithm interface: (log_probs, advantage, ...) -> loss.

Caller does: loss = algorithm.compute_loss(...); loss.backward();
algorithm.clip_grad_if_needed(parameters); optimizer.step();
Optional: algorithm.update_after_step(...) for e.g. baseline EMA or PPO epochs.
"""

from collections.abc import Iterable
from typing import Protocol

import torch


class Algorithm(Protocol):
    """Interface for policy-gradient algorithms: compute loss from log-probs and advantage."""

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        advantage: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Return scalar loss. Caller is responsible for backward and optimizer step."""
        ...

    def clip_grad_if_needed(self, parameters: Iterable[torch.Tensor]) -> None:
        """Optionally clip gradients after backward (no-op if algorithm has no grad clip)."""
        ...

    def update_after_step(self, **kwargs: object) -> None:
        """Optional hook after optimizer step (e.g. PPO multiple epochs). Default: no-op."""
        ...


class ReinforceAlgorithm:
    """REINFORCE: loss = mean(-sum(log_probs, dim=1) * advantage)."""

    def __init__(self, grad_clip: float | None = None) -> None:
        self.grad_clip = grad_clip

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        advantage: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        """Policy gradient loss: (-log_probs.sum(1) * advantage).mean()."""
        return (-torch.sum(log_probs, dim=1) * advantage).mean()

    def clip_grad_if_needed(self, parameters: Iterable[torch.Tensor]) -> None:
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)

    def update_after_step(self, **kwargs: object) -> None:
        pass

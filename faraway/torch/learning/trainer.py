"""
Trainer: orchestrates env + setting + rollout + advantage + algorithm per learning step.

One learning_step(): build_players → rollout → reward_from_scores → advantage
→ loss → step → log → eval.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from faraway.torch.env import BatchedFarawayEnv
from faraway.torch.learning.rollout import run_rollout

# Algorithm and advantage are typed as having the right methods (Protocol/duck typing)
# to avoid circular imports; concrete types come from learning_runner.


def _get_baseline(advantage: Any) -> float:
    return getattr(advantage, "baseline", 0.0)


class Trainer:
    """One step: build_players → rollout → reward → advantage → loss → step."""

    def __init__(
        self,
        env: BatchedFarawayEnv,
        setting: Any,
        advantage: Any,
        algorithm: Any,
        learner: Any,
        optimizer: torch.optim.Optimizer,
        batch_size: int,
        initial_temperature: float = 1.0,
        temperature_decay: float = 1.0,
        writer: SummaryWriter | None = None,
        verbose: int = 0,
        eval_flows: list[dict[str, Any]] | None = None,
        on_eval_flow: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.env = env
        self.setting = setting
        self.advantage = advantage
        self.algorithm = algorithm
        self.learner = learner
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.initial_temperature = initial_temperature
        self.temperature_decay = temperature_decay
        self.writer = writer
        self.verbose = verbose
        self.eval_flows = eval_flows or []
        self.on_eval_flow = on_eval_flow
        self.step_id = 0

    def learning_step(self) -> None:
        batch_size = self.batch_size
        temperature = max(
            1.0,
            self.initial_temperature * (self.temperature_decay**self.step_id),
        )

        players = self.setting.build_players(self.learner, self.env)
        result = run_rollout(
            self.env,
            players,
            learner_index=self.setting.learner_index,
            temperature=temperature,
            batch_size=batch_size,
        )

        scores_learner = result.scores[:, self.setting.learner_index]
        bonus_cards_played = self.env.get_bonus_cards_played(players)[:, self.setting.learner_index]
        main_card_ids = players[self.setting.learner_index].fields["main"][:, :, 0]

        reward = self.setting.reward_from_scores(
            scores_learner,
            bonus_cards_played=bonus_cards_played,
            main_card_ids=main_card_ids,
        )

        if result.env_state is not None:
            advantage = self.advantage.compute(
                reward,
                n_sub_batches=result.env_state["n_sub_batches"],
                current_sub_batch_size=result.env_state["current_sub_batch_size"],
            )
        else:
            advantage = self.advantage.compute(reward)

        loss = self.algorithm.compute_loss(result.log_probs_learner, advantage)
        self.optimizer.zero_grad()
        loss.backward()
        self.algorithm.clip_grad_if_needed(self.learner.model.parameters())
        self.optimizer.step()
        self.algorithm.update_after_step()
        self.advantage.update(reward)

        self.learner.n_training_games_played += batch_size
        n_training_games_played = self.learner.n_training_games_played

        if self.writer is not None:
            self.writer.add_scalar(
                "solo_train_score/mean", scores_learner.mean().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/max", scores_learner.max().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/min", scores_learner.min().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/std", scores_learner.std().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_bonus/mean",
                bonus_cards_played.mean().item(),
                n_training_games_played,
            )
            self.writer.add_scalar(
                "solo_train_bonus/max",
                bonus_cards_played.max().item(),
                n_training_games_played,
            )
            self.writer.add_scalar(
                "solo_train_avg_card_id/mean",
                main_card_ids.mean().item(),
                n_training_games_played,
            )
            id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]
            n_increases = id_diffs.float().sum(dim=1)
            self.writer.add_scalar(
                "solo_train_id_increases/mean",
                n_increases.mean().item(),
                n_training_games_played,
            )
            if (
                getattr(self.setting, "id_increase_reward_weight", 0) > 0
                or getattr(self.setting, "bonus_reward_weight", 0) > 0
                or getattr(self.setting, "low_id_reward_weight", 0) > 0
            ):
                self.writer.add_scalar(
                    "solo_train_shaped_reward/mean",
                    reward.mean().item(),
                    n_training_games_played,
                )
            self.writer.add_scalar(
                "baseline/value", _get_baseline(self.advantage), n_training_games_played
            )
            self.writer.add_scalar(
                "advantage/mean", advantage.mean().item(), n_training_games_played
            )
            self.writer.add_scalar("advantage/std", advantage.std().item(), n_training_games_played)
            self.writer.add_scalar("loss/policy", loss.item(), n_training_games_played)
            if self.initial_temperature > 1.0:
                self.writer.add_scalar("temperature/value", temperature, n_training_games_played)
            self.writer.flush()

        if self.verbose > 0:
            from loguru import logger

            logger.info(
                f"Step {self.step_id}. "
                f"Score: {scores_learner.mean().item():.2f}. "
                f"Baseline: {_get_baseline(self.advantage):.2f}. "
                f"Loss: {loss.item():.2f}. "
                f"Games: {n_training_games_played}"
            )

        self.step_id += 1

        for flow in self.eval_flows:
            every = flow.get("every", 500)
            initial_eval = flow.get("initial_eval", False)
            due = self.step_id % every == 0 or (self.step_id == 1 and initial_eval)
            if due and self.on_eval_flow is not None:
                self.on_eval_flow(flow)

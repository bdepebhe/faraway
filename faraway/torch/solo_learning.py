"""
Pure tensor-based solo play for Faraway game.

No dependencies on legacy pydantic classes - everything is tensors.

Tensor representations:
- Main card: 24 features [id, assets(9), rewards(11), prerequisites(3)]
- Bonus card: 20 features [assets(9), rewards(11)]
- Player main field: (batch, 8, 24)
- Player bonus field: (batch, 7, 20)
- Deck availability tracked via index masks
"""

import sys
from typing import Annotated, Any

import torch
import typer
from loguru import logger

from faraway.torch.base_game import BaseNNGame
from faraway.torch.env import BatchedFarawayEnv
from faraway.torch.learning import (
    BaselineEMA,
    PeerRelativeCenter,
    PeerRelativeZScore,
    run_rollout,
)
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.nn_player import BaseNNPlayer
from faraway.torch.play_vs_random import play_vs_random
from faraway.torch.transformers_player import TransformersPlayer


def sample_cards_from_availability_tensor(
    availability_tensor: torch.Tensor, draft_size: int
) -> torch.Tensor:
    # the availability tensor is a boolean tensor of shape (batch, n_cards)
    # we want to return a tensor of size (batch, draft_size) with the indices of the sampled cards
    # we can use torch.multinomial to sample the indices with replacement
    return torch.multinomial(availability_tensor, draft_size, replacement=False)


class SoloLearningGame(BaseNNGame):
    """
    Batched solo play game using tensors for REINFORCE-based training.

    Supports two advantage computation modes:
        - Baseline-based (default): advantage = reward - EMA_baseline
        - Peer-relative: advantage = (reward - batch_mean) / batch_std (z-score)
          or advantage = reward - batch_mean (center)

    Peer-relative reward mode (rl_params["peer_relative_reward"] = True):
        Games in a sub-batch see the exact same cards in the same order (fixed seed).
        This enables fair comparison: score differences are purely due to model decisions.
        Advantage is computed within each sub-batch (z-score or centering).
        Requires appropriate deck/draft sizes to avoid deck exhaustion (e.g., hand=3, draft=3).

        Sub-batches (rl_params["peer_sub_batch_size"]):
            The training batch can be divided into sub-batches. Each sub-batch gets a
            different deck shuffle, but games within the same sub-batch see identical cards.
            Example: batch_size=128, peer_sub_batch_size=32 → 4 sub-batches with different
            deck shuffles, advantage normalized within each 32-game sub-batch.
            If peer_sub_batch_size is None, the entire batch is one sub-batch.

        In this mode, prior_baseline_score and update_baseline_rate are NOT used for
        training (advantage comes from sub-batch mean), but the baseline EMA is still
        tracked for TensorBoard monitoring.

    Attributes:
        n_rounds: Number of rounds in the game
        draft_size: Number of cards shown in each draft
        peer_relative_reward: If True, use fixed seed and within-sub-batch advantage
        peer_sub_batch_size: Size of sub-batches for peer comparison (None = full batch)
        advantage_peer_normalization: "zscore" or "center" (only used if peer_relative_reward)
        device: Torch device for tensors

    Tensor shapes:
        main_field: (batch, 8, 24) - played main cards
        bonus_field: (batch, 6, 24) - played bonus cards
        field_state: (batch, 8+6, 24) - played cards
        field_state_flattened: (batch, (8+6)*24 + 1) - same flattened, + 1 for round index
        expanded_field_state: (batch, draft_size, (8+6)*24 + 1) - same, expanded for draft
        nn_input_tensor: (batch, draft_size, (8+6)*24 + 1 + 24) - same, + 24 for the possible card
        nn_logits: (batch, draft_size) - logits from the model
        nn_selected_index: (batch,) - index of the selected card from the possible cards
        nn_sampled_probability: (batch,) - probability of the selected card. used for training
        main_deck: (68, 24) - CONSTANT: full deck
        bonus_deck: (45, 24) - CONSTANT: bonus full deck
        main_deck_availability: (batch, 68): boolean masking
        bonus_deck_availability: (batch, 45): boolean masking
    """

    def __init__(
        self,
        n_rounds: int = 8,
        draft_size: int = 10,
        replace_remaining_cards: bool = True,
        use_bonus_cards: bool = True,
        use_hand: bool = False,
        n_cards_hand: int = 3,
        model_path: str | None = None,
        verbose: int = 1,
        device: torch.device | None = None,
        model_params: dict[str, Any] | None = None,
        player_type: str = "mlp",
        player_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        rl_params: dict[str, Any] | None = None,
        experiment_name: str | None = None,
        log_dir: str = "runs",
        eval_vs_random_config: dict[str, Any] | None = None,
        eval_solo_config: dict[str, Any] | None = None,
    ):
        super().__init__(
            n_rounds,
            use_bonus_cards,
            device,
            verbose=verbose,
            experiment_name=experiment_name,
            log_dir=log_dir,
        )
        self.draft_size = draft_size
        self.replace_remaining_cards = replace_remaining_cards
        self.use_hand = use_hand
        self.n_cards_hand = n_cards_hand
        self.model_params = model_params or {
            "hidden_layers_sizes": [512, 512],
            "dropout_rate": 0.1,
        }
        self.player_type = player_type
        self.player_params = player_params or {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
        self.optimizer_params = optimizer_params or {
            "lr": 0.001,
        }
        self.rl_params = rl_params or {
            "prior_baseline_score": 29,
            "train_batch_size": 32,
            "update_baseline_rate": 0.05,
        }
        # Peer-relative reward settings
        self.peer_relative_reward = self.rl_params.get("peer_relative_reward", False)
        self.advantage_peer_normalization = self.rl_params.get(
            "advantage_peer_normalization", "zscore"
        )  # "zscore" or "center"

        # Temperature annealing settings (for exploration)
        # temperature = max(1.0, initial_temperature * temperature_decay^step)
        self.initial_temperature = self.rl_params.get("initial_temperature", 1.0)
        self.temperature_decay = self.rl_params.get("temperature_decay", 1.0)
        self.current_temperature = self.initial_temperature  # will be updated each step
        # Sub-batch size for peer comparison (games within a sub-batch share the same deck shuffle)
        # If None or 0, the entire batch is one sub-batch (all games see same cards)
        self.peer_sub_batch_size = self.rl_params.get("peer_sub_batch_size", None)
        self.player_params["use_bonus_cards"] = self.use_bonus_cards

        # Advantage: baseline EMA (always used for tracking; also for adv when not peer-relative)
        self._baseline_ema = BaselineEMA(
            prior_baseline=self.rl_params["prior_baseline_score"],
            update_rate=self.rl_params["update_baseline_rate"],
        )
        self.baseline = self._baseline_ema.baseline
        # Peer-relative advantage (only used when peer_relative_reward=True)
        norm = self.advantage_peer_normalization
        self._advantage_peer = PeerRelativeZScore() if norm == "zscore" else PeerRelativeCenter()

        # Evaluation config
        self.eval_vs_random_config = eval_vs_random_config or {}
        self.eval_solo_config = eval_solo_config or {}

        # Common batched environment (Phase B refactor)
        self._env = BatchedFarawayEnv(
            n_rounds=self.n_rounds,
            use_bonus_cards=self.use_bonus_cards,
            draft_size=self.draft_size,
            use_hand=self.use_hand,
            n_cards_hand=self.n_cards_hand,
            replace_remaining_cards=self.replace_remaining_cards,
            device=self.device,
            peer_relative_reward=self.peer_relative_reward,
            peer_sub_batch_size=self.peer_sub_batch_size,
            verbose=self.verbose,
        )
        # Initialize TensorBoard (uses base class method)
        self.init_tensorboard()
        self.reset_learning(model_path=model_path)
        self.players: list[BaseNNPlayer]

    def reset_learning(self, model_path: str | None = None) -> None:
        if model_path is not None:
            model = torch.load(model_path)
        else:
            model = None
        self._baseline_ema.baseline = self.rl_params["prior_baseline_score"]
        self.baseline = self._baseline_ema.baseline
        self.step_id = 0  # step counter for evaluation frequency (session-local, not saved)

        if self.player_type == "mlp":
            self.players = [
                MLPPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]  # only one player for solo play
        elif self.player_type == "transformer":
            self.players = [
                TransformersPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]  # only one player for solo play
        else:
            raise ValueError(f"Unknown player type: {self.player_type}")
        self.optimizer = torch.optim.Adam(
            self.players[0].model.parameters(), **self.optimizer_params
        )

    def reset_games_batch(self, batch_size: int) -> None:
        """Reset games and initialize player hands via common env."""
        self._env.reset(batch_size, self.players)
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index

    def get_scores(self) -> torch.Tensor:
        """Compute final scores via common env."""
        return self._env.get_scores(self.players)

    def get_bonus_cards_played(self) -> torch.Tensor:
        """Count bonus cards played per player via common env."""
        return self._env.get_bonus_cards_played(self.players)

    def dump_model(self, model_path: str) -> None:
        torch.save(self.players[0].model, model_path)

    def dump_player(self, player_path: str) -> None:
        """Save the player (model + config) to a file."""
        self.players[0].dump(player_path)

    def dump_training_state(self, path: str) -> None:
        """Save training state (player + baseline) to a file."""
        checkpoint = {
            "player_state": self.players[0].model.state_dict(),
            "player_params": getattr(self.players[0], "model_params", {}),
            "player_config": {
                "n_rounds": self.players[0].n_rounds,
                "use_bonus_cards": self.players[0].use_bonus_cards,
                "n_cards_hand": self.players[0].n_cards_hand,
            },
            "n_training_games_played": self.players[0].n_training_games_played,
            "baseline": self.baseline,
            "player_type": self.player_type,
        }
        # Add transformer-specific config
        if hasattr(self.players[0], "use_mode_embedding"):
            checkpoint["player_config"]["use_mode_embedding"] = self.players[0].use_mode_embedding
        torch.save(checkpoint, path)

    def load_training_state(self, path: str) -> None:
        """Load training state (model weights + baseline) from a file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.players[0].model.load_state_dict(checkpoint["player_state"])
        self.players[0].n_training_games_played = checkpoint.get("n_training_games_played", 0)
        if "baseline" in checkpoint:
            self._baseline_ema.baseline = checkpoint["baseline"]
            self.baseline = self._baseline_ema.baseline
            logger.info(f"Restored baseline: {self.baseline:.2f}")
        logger.info(f"Restored n_training_games_played: {self.players[0].n_training_games_played}")

    def run_eval_vs_random(self) -> tuple[float, float]:
        """Run evaluation against random players using the shared TensorBoard writer."""
        n_random_players = self.eval_vs_random_config.get("n_players", 1)

        if self.verbose > 0:
            logger.info(f"Running eval vs {n_random_players} random player(s)...")
        win_rate, mean_score = play_vs_random(
            player=self.players[0],
            n_random_players=n_random_players,
            n_eval_batches=self.eval_vs_random_config.get("n_batches", 100),
            batch_size=self.eval_vs_random_config.get("batch_size", 32),
            writer=self.writer,  # Share the TensorBoard writer
            verbose=0,  # Quiet mode for intermediate evals
        )
        if self.verbose > 0:
            logger.info(f"Eval vs random: win_rate={win_rate:.2%}, mean_score={mean_score:.2f}")
        return win_rate, mean_score

    def run_eval_solo(self) -> float:
        """Run solo evaluation and log to TensorBoard."""
        n_batches = self.eval_solo_config.get("n_batches", 100)
        batch_size = self.eval_solo_config.get("batch_size", 32)

        if self.verbose > 0:
            logger.info(f"Running solo eval ({n_batches} batches x {batch_size})...")

        scores = self.play_games_batches(n_batches=n_batches, batch_size=batch_size)
        mean_score = scores.mean().item()

        if self.writer is not None:
            step = self.players[0].n_training_games_played
            self.writer.add_scalar("eval/solo/mean_score", mean_score, step)
            self.writer.add_scalar("eval/solo/max_score", scores.max().item(), step)
            self.writer.add_scalar("eval/solo/min_score", scores.min().item(), step)
            self.writer.add_scalar("eval/solo/std_score", scores.std().item(), step)
            self.writer.flush()

        if self.verbose > 0:
            logger.info(f"Eval solo: mean_score={mean_score:.2f}")

        return float(mean_score)

    def initialize_baseline_from_eval(
        self,
        n_batches: int | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Initialize baseline from solo evaluation score.

        This runs a solo evaluation and sets the baseline to the mean score,
        providing a more accurate initial baseline than a hardcoded value.

        Args:
            n_batches: Number of evaluation batches (default: from eval_solo_config or 50)
            batch_size: Games per batch (default: from eval_solo_config or 64)

        Returns:
            The mean score used as the new baseline.
        """
        # Use provided values or fall back to eval_solo_config or defaults
        n_batches = n_batches or self.eval_solo_config.get("n_batches", 50)
        batch_size = batch_size or self.eval_solo_config.get("batch_size", 64)

        if self.verbose > 0:
            logger.info(
                f"Initializing baseline from solo eval ({n_batches} batches x {batch_size})..."
            )

        # Run evaluation without TensorBoard logging (just for baseline init)
        scores = self.play_games_batches(n_batches=n_batches, batch_size=batch_size)
        mean_score = scores.mean().item()

        # Set baseline
        old_baseline = self.baseline
        self.baseline = mean_score

        if self.verbose > 0:
            logger.info(
                f"Baseline initialized: {old_baseline:.2f} -> {self.baseline:.2f} "
                f"(from {n_batches * batch_size} games)"
            )

        return float(mean_score)

    def log_hparams(self, extra_hparams: dict[str, Any] | None = None) -> None:
        """Log hyperparameters to TensorBoard for experiment comparison."""
        # Count trainable parameters
        n_trainable_params = sum(
            p.numel() for p in self.players[0].model.parameters() if p.requires_grad
        )

        hparams = {
            "player_type": self.player_type,
            "n_rounds": self.n_rounds,
            "draft_size": self.draft_size,
            "use_hand": self.use_hand,
            "n_cards_hand": self.n_cards_hand,
            "n_trainable_params": n_trainable_params,
            "rl_params": self.rl_params,
            "model_params": self.model_params,
            "player_params": self.player_params,
            "optimizer_params": self.optimizer_params,
            "use_bonus_cards": self.use_bonus_cards,
            "replace_remaining_cards": self.replace_remaining_cards,
        }
        if extra_hparams:
            hparams.update(extra_hparams)
        if self.writer is not None:
            # Log trainable params as a scalar for easy comparison across runs
            self.writer.add_scalar("model/n_trainable_params", n_trainable_params, 0)
            # Use add_text instead of add_hparams to avoid creating timestamp subdirectories
            hparams_text = "\n".join(f"**{k}**: {v}" for k, v in hparams.items())
            self.writer.add_text("hparams", hparams_text, 0)

    def play_round(self) -> torch.Tensor:
        """Play one round via common env. Returns probabilities of all decisions this round."""
        result = self._env.step_round(self.players, temperature=self.current_temperature)
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index
        return result.picked_probabilities[0]

    def learning_step(self) -> None:
        batch_size = self.rl_params["train_batch_size"]

        # Compute current temperature with exponential decay (clamped to minimum of 1.0)
        self.current_temperature = max(
            1.0, self.initial_temperature * (self.temperature_decay**self.step_id)
        )

        # Rollout: run env for one batch; collect log-probs and scores
        result = run_rollout(
            self._env,
            self.players,
            learner_index=0,
            temperature=self.current_temperature,
            batch_size=batch_size,
        )
        log_probs = result.log_probs_learner  # (batch, n_decisions)
        scores = result.scores[:, 0]

        # Reward shaping: Count how many times the model played increasing IDs
        # This directly rewards the BEHAVIOR that leads to bonus cards, not the
        # delayed outcome. More direct credit assignment.
        # Normalized to [-1, 1]: 0 increases = -1, 3.5 increases = 0, 7 increases = +1
        id_increase_reward_weight = self.rl_params.get("id_increase_reward_weight", 0.0)
        max_increases = self.n_rounds - 1  # 7 possible increases for 8 rounds
        mid_increases = max_increases / 2.0  # 3.5 = random expectation
        if id_increase_reward_weight > 0:
            player = self.players[0]
            main_card_ids = player.fields["main"][:, :, 0]  # (batch, 8)
            # Count ID increases: id[i+1] > id[i]
            id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]  # (batch, 7) bool
            n_increases = id_diffs.float().sum(dim=1)  # (batch,)
            # Normalize to [-1, 1]
            increases_normalized = (n_increases - mid_increases) / mid_increases
            shaped_reward = scores + id_increase_reward_weight * increases_normalized
        else:
            n_increases = None
            shaped_reward = scores

        # DEPRECATED: bonus_reward_weight - use id_increase_reward_weight instead
        # The bonus card count is an indirect signal; id_increase is the direct behavior
        bonus_reward_weight = self.rl_params.get("bonus_reward_weight", 0.0)
        max_bonus_cards = self.n_rounds - 1  # 7 for 8 rounds
        mid_bonus_cards = max_bonus_cards / 2.0  # 3.5 = random average
        if bonus_reward_weight > 0 and self.use_bonus_cards:
            bonus_cards_played = self.get_bonus_cards_played()[:, 0]  # (batch,)
            bonus_normalized = (bonus_cards_played - mid_bonus_cards) / mid_bonus_cards
            shaped_reward = shaped_reward + bonus_reward_weight * bonus_normalized
        else:
            bonus_cards_played = None

        # Reward shaping: optionally reward playing lower card IDs
        # WARNING: This conflicts with id_increase_reward! Low IDs make increasing harder.
        # Only use this if you specifically need "draft first" behavior over bonus cards.
        # Normalized to [-1, 1]: -1 = ID 68, 0 = ID 34.5 (middle), +1 = ID 1
        low_id_reward_weight = self.rl_params.get("low_id_reward_weight", 0.0)
        if low_id_reward_weight > 0:
            player = self.players[0]
            main_card_ids = player.fields["main"][:, :, 0]  # (batch, 8)
            avg_card_id = main_card_ids.mean(dim=1)  # (batch,)
            max_card_id = 68.0  # Cards are numbered 1-68
            mid_card_id = max_card_id / 2.0  # 34 = middle
            low_id_normalized = (mid_card_id - avg_card_id) / mid_card_id
            shaped_reward = shaped_reward + low_id_reward_weight * low_id_normalized

        # Compute advantage: peer-relative or baseline-based
        if self.peer_relative_reward and result.env_state is not None:
            advantage = self._advantage_peer.compute(
                shaped_reward,
                result.env_state["n_sub_batches"],
                result.env_state["current_sub_batch_size"],
            )
        else:
            advantage = self._baseline_ema.compute(shaped_reward)

        loss = (-torch.sum(log_probs, 1) * advantage).mean()  # scalar
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (especially with mode embedding)
        if self.rl_params.get("grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.players[0].model.parameters(), self.rl_params["grad_clip"]
            )
        self.optimizer.step()

        # Update total games played (epoch metric based on environment interactions)
        self.players[0].n_training_games_played += self.rl_params["train_batch_size"]
        n_training_games_played = self.players[0].n_training_games_played

        # Log metrics to TensorBoard
        # Using n_training_games_played as x-axis for fair comparison across batch sizes
        if self.writer is not None:
            self.writer.add_scalar(
                "solo_train_score/mean", scores.mean().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/max", scores.max().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/min", scores.min().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/std", scores.std().item(), n_training_games_played
            )
            # Log bonus cards played (max 7 for 8 rounds)
            if self.use_bonus_cards:
                # Reuse bonus_cards_played if already computed for reward shaping
                if bonus_cards_played is None:
                    bonus_cards_played = self.get_bonus_cards_played()[:, 0]  # (batch,)
                self.writer.add_scalar(
                    "solo_train_bonus/mean",
                    bonus_cards_played.mean().item(),
                    n_training_games_played,
                )
                self.writer.add_scalar(
                    "solo_train_bonus/max", bonus_cards_played.max().item(), n_training_games_played
                )
            # Log average card ID (lower = better for draft priority)
            main_card_ids = self.players[0].fields["main"][:, :, 0]  # (batch, 8)
            self.writer.add_scalar(
                "solo_train_avg_card_id/mean", main_card_ids.mean().item(), n_training_games_played
            )
            # Log ID increases (key metric for bonus card acquisition behavior)
            if n_increases is None:
                id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]
                n_increases = id_diffs.float().sum(dim=1)
            self.writer.add_scalar(
                "solo_train_id_increases/mean", n_increases.mean().item(), n_training_games_played
            )
            # Log shaped reward if using any reward shaping
            if id_increase_reward_weight > 0 or bonus_reward_weight > 0 or low_id_reward_weight > 0:
                self.writer.add_scalar(
                    "solo_train_shaped_reward/mean",
                    shaped_reward.mean().item(),
                    n_training_games_played,
                )
            self.writer.add_scalar("baseline/value", self.baseline, n_training_games_played)
            self.writer.add_scalar(
                "advantage/mean", advantage.mean().item(), n_training_games_played
            )
            self.writer.add_scalar("advantage/std", advantage.std().item(), n_training_games_played)
            self.writer.add_scalar("loss/policy", loss.item(), n_training_games_played)
            # Log temperature if using annealing
            if self.initial_temperature > 1.0:
                self.writer.add_scalar(
                    "temperature/value", self.current_temperature, n_training_games_played
                )

        if self.verbose > 0:
            logger.info(
                f"Step {self.step_id}. "
                f"Score: {scores.mean().item():.2f}. "
                f"Baseline: {self.baseline:.2f}. "
                f"Loss: {loss.item():.2f}. "
                f"Games: {n_training_games_played}"
            )
        # Update baseline EMA (for advantage when not peer-relative; always for monitoring)
        self._baseline_ema.update(shaped_reward)
        self.baseline = self._baseline_ema.baseline
        self.step_id += 1

        # Run periodic evaluations
        if self.eval_vs_random_config and (
            self.step_id % self.eval_vs_random_config.get("every", 500) == 0
            or (self.step_id == 1 and self.eval_vs_random_config.get("initial_eval", False))
        ):
            self.run_eval_vs_random()

        if self.eval_solo_config and (
            self.step_id % self.eval_solo_config.get("every", 500) == 0
            or (self.step_id == 1 and self.eval_solo_config.get("initial_eval", False))
        ):
            self.run_eval_solo()


def main(
    log_to_file: Annotated[bool, typer.Option(help="Whether to log to a file")] = False,
    experiment_name: Annotated[
        str | None, typer.Option(help="Name for TensorBoard experiment")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 32,
    draft_size: Annotated[int, typer.Option(help="Draft size")] = 10,
    n_steps: Annotated[int, typer.Option(help="Number of training steps")] = 1000,
    n_eval_batches: Annotated[int, typer.Option(help="Number of evaluation batches")] = 100,
    eval_vs_random_every: Annotated[
        int | None, typer.Option(help="Run eval vs random every N steps (None to disable)")
    ] = None,
    eval_vs_random_n_players: Annotated[
        int, typer.Option(help="Number of random players for eval")
    ] = 1,
    eval_solo_every: Annotated[
        int | None, typer.Option(help="Run eval solo every N steps (None to disable)")
    ] = None,
    player_type: Annotated[str, typer.Option(help="Player type: 'mlp' or 'transformer'")] = "mlp",
    use_mode_embedding: Annotated[
        bool, typer.Option(help="Use mode embedding for transformer (play/draft/bonus)")
    ] = False,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 0.0005,
    baseline_update_rate: Annotated[
        float, typer.Option(help="Baseline EMA update rate (lower = smoother)")
    ] = 0.05,
    grad_clip: Annotated[
        float | None, typer.Option(help="Gradient clipping max norm (None to disable)")
    ] = None,
    use_hand: Annotated[
        bool, typer.Option(help="Enable draft mechanism (hand management)")
    ] = False,
    n_cards_hand: Annotated[
        int, typer.Option(help="Number of cards in hand (when using draft)")
    ] = 3,
    peer_relative_reward: Annotated[
        bool, typer.Option(help="Use peer-relative reward (same deck for all games in batch)")
    ] = False,
    advantage_peer_normalization: Annotated[
        str, typer.Option(help="Peer-relative mode: 'zscore' or 'center'")
    ] = "zscore",
    initial_temperature: Annotated[
        float,
        typer.Option(help="Initial softmax temperature for exploration (>1 = more exploration)"),
    ] = 1.0,
    temperature_decay: Annotated[
        float, typer.Option(help="Temperature decay rate per step (e.g., 0.9999)")
    ] = 1.0,
) -> None:
    """Run a solo learning game."""
    logger.remove()  # remove default stderr handler
    if log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    eval_vs_random_config: dict[str, Any] | None = None
    if eval_vs_random_every is not None:
        eval_vs_random_config = {
            "every": eval_vs_random_every,
            "n_players": eval_vs_random_n_players,
            "n_batches": n_eval_batches,
            "batch_size": batch_size,
            "initial_eval": True,
        }

    eval_solo_config: dict[str, Any] | None = None
    if eval_solo_every is not None:
        eval_solo_config = {
            "every": eval_solo_every,
            "n_batches": n_eval_batches,
            "batch_size": batch_size,
            "initial_eval": True,
        }

    # Choose model params based on player type
    if player_type == "mlp":
        model_params = {
            "hidden_layers_sizes": [512, 512],
            "dropout_rate": 0.1,
        }
        player_params = {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
    elif player_type == "transformer":
        model_params = {
            "embed_dim": 64,  # 64,
            "n_attention_heads": 4,  # 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }
        player_params = {
            "use_mode_embedding": use_mode_embedding,
        }
    else:
        raise ValueError(f"Unknown player type: {player_type}")

    rl_params: dict[str, Any] = {
        "prior_baseline_score": 29,
        "train_batch_size": batch_size,
        "update_baseline_rate": baseline_update_rate,
        "peer_relative_reward": peer_relative_reward,
        "advantage_peer_normalization": advantage_peer_normalization,
        "initial_temperature": initial_temperature,
        "temperature_decay": temperature_decay,
    }
    if grad_clip is not None:
        rl_params["grad_clip"] = grad_clip

    game = SoloLearningGame(
        verbose=2,
        experiment_name=experiment_name,
        model_params=model_params,
        player_params=player_params,
        player_type=player_type,
        optimizer_params={
            "lr": lr,
        },
        rl_params=rl_params,
        draft_size=draft_size,
        use_hand=use_hand,
        n_cards_hand=n_cards_hand,
        eval_vs_random_config=eval_vs_random_config,
        eval_solo_config=eval_solo_config,
    )

    # Log hyperparameters for experiment comparison
    game.log_hparams({"n_steps": n_steps})

    # Training
    for _ in range(n_steps):
        game.learning_step()

    game.dump_player(f"runs/{game.experiment_name}/player.pt")
    game.close_tensorboard()
    print(f"\nTensorBoard logs saved to: runs/{game.experiment_name}")
    print("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

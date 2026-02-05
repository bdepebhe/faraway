"""
Setting-agnostic learning runner: env + setting + advantage + algorithm + trainer.

Default setting is SoloSetting (solo play with optional reward shaping). Pass a different
setting (e.g. VsRandomSetting, SelfPlaySetting) to run other modes. Same API for
run_experiment, eval, and CLI.
"""

import sys
from typing import Annotated, Any

import torch
import typer
from loguru import logger

from faraway.torch.base_game import BaseNNGame
from faraway.torch.env import BatchedFarawayEnv
from faraway.torch.learning import (
    AdvantageWithBaselineTracking,
    BaselineEMA,
    PeerRelativeCenter,
    PeerRelativeZScore,
    ReinforceAlgorithm,
    SoloSetting,
    Trainer,
)
from faraway.torch.learning.settings import Setting
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.nn_player import BaseNNPlayer
from faraway.torch.play_vs_random import play_vs_random
from faraway.torch.transformers_player import TransformersPlayer


def sample_cards_from_availability_tensor(
    availability_tensor: torch.Tensor, draft_size: int
) -> torch.Tensor:
    """Sample draft_size indices from availability (batch, n_cards). -> (batch, draft_size)."""
    return torch.multinomial(availability_tensor, draft_size, replacement=False)


class LearningRunner(BaseNNGame):
    """
    Setting-agnostic learning runner: env + setting + advantage + algorithm + trainer.

    Use default setting=None for solo play (SoloSetting with reward shaping from rl_params).
    Pass a Setting implementation (e.g. VsRandomSetting in Phase F) to run other modes.
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
        setting: Setting | None = None,
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
        self.peer_relative_reward = self.rl_params.get("peer_relative_reward", False)
        self.advantage_peer_normalization = self.rl_params.get(
            "advantage_peer_normalization", "zscore"
        )
        self.initial_temperature = self.rl_params.get("initial_temperature", 1.0)
        self.temperature_decay = self.rl_params.get("temperature_decay", 1.0)
        self.current_temperature = self.initial_temperature
        self.peer_sub_batch_size = self.rl_params.get("peer_sub_batch_size", None)
        self.player_params["use_bonus_cards"] = self.use_bonus_cards

        self._baseline_ema = BaselineEMA(
            prior_baseline=self.rl_params["prior_baseline_score"],
            update_rate=self.rl_params["update_baseline_rate"],
        )
        self.baseline = self._baseline_ema.baseline
        norm = self.advantage_peer_normalization
        self._advantage_peer = PeerRelativeZScore() if norm == "zscore" else PeerRelativeCenter()
        self._algorithm = ReinforceAlgorithm(grad_clip=self.rl_params.get("grad_clip"))
        self._setting = (
            setting
            if setting is not None
            else SoloSetting(
                n_rounds=self.n_rounds,
                use_bonus_cards=self.use_bonus_cards,
                id_increase_reward_weight=self.rl_params.get("id_increase_reward_weight", 0.0),
                bonus_reward_weight=self.rl_params.get("bonus_reward_weight", 0.0),
                low_id_reward_weight=self.rl_params.get("low_id_reward_weight", 0.0),
            )
        )
        self._advantage = AdvantageWithBaselineTracking(
            compute_strategy=(
                self._advantage_peer if self.peer_relative_reward else self._baseline_ema
            ),
            baseline_ema=self._baseline_ema,
        )

        self.eval_vs_random_config = eval_vs_random_config or {}
        self.eval_solo_config = eval_solo_config or {}

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
        self.step_id = 0

        if self.player_type == "mlp":
            self.players = [
                MLPPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]
        elif self.player_type == "transformer":
            self.players = [
                TransformersPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]
        else:
            raise ValueError(f"Unknown player type: {self.player_type}")
        self.optimizer = torch.optim.Adam(
            self.players[0].model.parameters(), **self.optimizer_params
        )
        self._trainer = Trainer(
            env=self._env,
            setting=self._setting,
            advantage=self._advantage,
            algorithm=self._algorithm,
            learner=self.players[0],
            optimizer=self.optimizer,
            batch_size=self.rl_params["train_batch_size"],
            initial_temperature=self.initial_temperature,
            temperature_decay=self.temperature_decay,
            writer=self.writer,
            verbose=self.verbose,
            eval_vs_random_config=self.eval_vs_random_config or None,
            eval_solo_config=self.eval_solo_config or None,
            on_eval_vs_random=self.run_eval_vs_random,
            on_eval_solo=self.run_eval_solo,
        )
        self._trainer.step_id = 0

    def reset_games_batch(self, batch_size: int) -> None:
        self._env.reset(batch_size, self.players)
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index

    def get_scores(self) -> torch.Tensor:
        return self._env.get_scores(self.players)

    def get_bonus_cards_played(self) -> torch.Tensor:
        return self._env.get_bonus_cards_played(self.players)

    def dump_model(self, model_path: str) -> None:
        torch.save(self.players[0].model, model_path)

    def dump_player(self, player_path: str) -> None:
        self.players[0].dump(player_path)

    def dump_training_state(self, path: str) -> None:
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
        if hasattr(self.players[0], "use_mode_embedding"):
            checkpoint["player_config"]["use_mode_embedding"] = self.players[0].use_mode_embedding
        torch.save(checkpoint, path)

    def load_training_state(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.players[0].model.load_state_dict(checkpoint["player_state"])
        self.players[0].n_training_games_played = checkpoint.get("n_training_games_played", 0)
        if "baseline" in checkpoint:
            self._baseline_ema.baseline = checkpoint["baseline"]
            self.baseline = self._baseline_ema.baseline
            logger.info(f"Restored baseline: {self.baseline:.2f}")
        logger.info(f"Restored n_training_games_played: {self.players[0].n_training_games_played}")

    def run_eval_vs_random(self) -> tuple[float, float]:
        n_random_players = self.eval_vs_random_config.get("n_players", 1)
        if self.verbose > 0:
            logger.info(f"Running eval vs {n_random_players} random player(s)...")
        win_rate, mean_score = play_vs_random(
            player=self.players[0],
            n_random_players=n_random_players,
            n_eval_batches=self.eval_vs_random_config.get("n_batches", 100),
            batch_size=self.eval_vs_random_config.get("batch_size", 32),
            writer=self.writer,
            verbose=0,
        )
        if self.verbose > 0:
            logger.info(f"Eval vs random: win_rate={win_rate:.2%}, mean_score={mean_score:.2f}")
        return win_rate, mean_score

    def run_eval_solo(self) -> float:
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
        n_batches = n_batches or self.eval_solo_config.get("n_batches", 50)
        batch_size = batch_size or self.eval_solo_config.get("batch_size", 64)
        if self.verbose > 0:
            logger.info(
                f"Initializing baseline from solo eval ({n_batches} batches x {batch_size})..."
            )
        scores = self.play_games_batches(n_batches=n_batches, batch_size=batch_size)
        mean_score = scores.mean().item()
        old_baseline = self.baseline
        self.baseline = mean_score
        self._baseline_ema.baseline = mean_score
        if self.verbose > 0:
            logger.info(
                f"Baseline initialized: {old_baseline:.2f} -> {self.baseline:.2f} "
                f"(from {n_batches * batch_size} games)"
            )
        return float(mean_score)

    def log_hparams(self, extra_hparams: dict[str, Any] | None = None) -> None:
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
            self.writer.add_scalar("model/n_trainable_params", n_trainable_params, 0)
            hparams_text = "\n".join(f"**{k}**: {v}" for k, v in hparams.items())
            self.writer.add_text("hparams", hparams_text, 0)

    def play_round(self) -> torch.Tensor:
        result = self._env.step_round(self.players, temperature=self.current_temperature)
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index
        return result.picked_probabilities[0]

    def learning_step(self) -> None:
        self._trainer.learning_step()
        self.baseline = self._baseline_ema.baseline
        self.step_id = self._trainer.step_id
        self.current_temperature = max(
            1.0,
            self.initial_temperature * (self.temperature_decay**self.step_id),
        )


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
    """Run training with default solo setting (CLI)."""
    logger.remove()
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

    if player_type == "mlp":
        model_params = {"hidden_layers_sizes": [512, 512], "dropout_rate": 0.1}
        player_params = {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
    elif player_type == "transformer":
        model_params = {
            "embed_dim": 64,
            "n_attention_heads": 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }
        player_params = {"use_mode_embedding": use_mode_embedding}
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

    runner = LearningRunner(
        verbose=2,
        experiment_name=experiment_name,
        model_params=model_params,
        player_params=player_params,
        player_type=player_type,
        optimizer_params={"lr": lr},
        rl_params=rl_params,
        draft_size=draft_size,
        use_hand=use_hand,
        n_cards_hand=n_cards_hand,
        eval_vs_random_config=eval_vs_random_config,
        eval_solo_config=eval_solo_config,
    )
    runner.log_hparams({"n_steps": n_steps})
    for _ in range(n_steps):
        runner.learning_step()
    runner.dump_player(f"runs/{runner.experiment_name}/player.pt")
    runner.close_tensorboard()
    print(f"\nTensorBoard logs saved to: runs/{runner.experiment_name}")
    print("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

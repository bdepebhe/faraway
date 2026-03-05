"""
Config-driven training script for Faraway RL agents.

Supports:
- YAML/JSON configuration files
- Curriculum learning with multiple phases
- Checkpoint loading/saving between phases
- Flexible hyperparameter specification

Usage:
    python -m faraway.torch.main config.yaml
    python -m faraway.torch.main config.json --phase 2
"""

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml  # type: ignore[import-untyped]
from loguru import logger

from faraway.torch.config_loader import resolve_phase_config
from faraway.torch.learning_runner import LearningRunner
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.transformers_player import TransformersPlayer

# Keys from resolved phase config that are passed to LearningRunner(__init__)
_LEARNING_RUNNER_KEYS = frozenset(
    {
        "n_rounds",
        "draft_size",
        "replace_remaining_cards",
        "use_bonus_cards",
        "use_hand",
        "n_cards_hand",
        "verbose",
        "model_params",
        "player_type",
        "player_params",
        "rl_params",
        "eval_flows",
    }
)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            result: dict[str, Any] = yaml.safe_load(f)
            return result
        elif path.suffix == ".json":
            result = json.load(f)
            return result
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def run_phase(
    phase_config: dict[str, Any],
    phase_idx: int,
    base_experiment_name: str,
    log_dir: str,
    previous_checkpoint: str | None = None,
    resume_from: str | None = None,
) -> str:
    """Run a single training phase. Config must already be resolved
    (e.g. via resolve_phase_config)."""
    phase_name = phase_config.get("name", f"phase_{phase_idx}")
    experiment_name = f"{base_experiment_name}/{phase_name}"
    n_steps = phase_config["n_steps"]

    logger.info("=" * 60)
    logger.info(f"Starting Phase {phase_idx}: {phase_name}")
    logger.info("=" * 60)

    if resume_from:
        load_from: str | None = resume_from
        logger.info("Resuming training from checkpoint (--resume mode)")
    else:
        load_from = phase_config.get("load_from")
        if load_from == "previous" and previous_checkpoint:
            load_from = previous_checkpoint
        elif load_from == "previous":
            load_from = None

    runner_kwargs = {k: phase_config[k] for k in _LEARNING_RUNNER_KEYS if k in phase_config}
    runner_kwargs["experiment_name"] = experiment_name
    runner_kwargs["log_dir"] = log_dir

    adv = phase_config.get("rl_params", {}).get("advantage", {})
    logger.info(f"Player type: {phase_config.get('player_type')}")
    logger.info(f"Model params: {phase_config.get('model_params')}")
    logger.info(f"Player params: {phase_config.get('player_params')}")
    logger.info(f"RL params: {phase_config.get('rl_params')}")
    logger.info(f"Use draft: {phase_config.get('use_hand')}")
    logger.info(f"N steps: {n_steps}")
    if adv.get("peer_relative_reward"):
        sub_batch_size = adv.get("peer_sub_batch_size")
        sub_batch_info = f", sub_batch_size={sub_batch_size}" if sub_batch_size else ""
        logger.info(
            f"Peer-relative reward: enabled "
            f"(normalization={adv.get('advantage_peer_normalization', 'zscore')}{sub_batch_info})"
        )
    if load_from:
        logger.info(f"Loading from: {load_from}")

    runner = LearningRunner(**runner_kwargs)

    # Log trainable parameters count
    n_trainable_params = sum(
        p.numel() for p in runner.players[0].model.parameters() if p.requires_grad
    )
    logger.info(f"Trainable parameters: {n_trainable_params:,}")

    if load_from:
        logger.info(f"Loading training state from: {load_from}")
        try:
            runner.load_training_state(load_from)
            logger.info(
                f"Loaded training state: {runner.players[0].n_training_games_played} games, "
                f"baseline={runner.baseline:.2f}"
            )
        except KeyError:
            logger.info("Falling back to player-only checkpoint (no baseline)")
            player_type = phase_config.get("player_type", "mlp")
            loaded_player: MLPPlayer | TransformersPlayer
            if player_type == "mlp":
                loaded_player = MLPPlayer.load(load_from, device=runner.device)
            elif player_type == "transformer":
                loaded_player = TransformersPlayer.load(load_from, device=runner.device)
            else:
                raise ValueError(f"Unknown player type: {player_type}") from None
            runner.players[0].model.load_state_dict(loaded_player.model.state_dict())
            runner.players[0].n_training_games_played = loaded_player.n_training_games_played
            logger.info(f"Loaded player with {loaded_player.n_training_games_played} games played")

    initial_baseline = phase_config.get("initial_baseline")
    if initial_baseline is not None:
        if initial_baseline == "previous":
            # Keep the baseline from checkpoint (already loaded above)
            logger.info(f"Keeping baseline from checkpoint: {runner.baseline:.2f}")
        elif initial_baseline == "from_eval":
            # Run solo evaluation and use mean score as baseline
            runner.initialize_baseline_from_eval()
        elif isinstance(initial_baseline, int | float):
            runner.baseline = float(initial_baseline)
            logger.info(f"Set initial baseline to: {runner.baseline:.2f}")
        else:
            raise ValueError(
                f"Invalid initial_baseline value: {initial_baseline}. "
                "Expected 'previous', 'from_eval', or a number."
            )
    elif not load_from:
        # No checkpoint loaded and no initial_baseline specified - use rl_params default
        logger.info(f"Using default baseline: {runner.baseline:.2f}")

    # Log hyperparameters
    runner.log_hparams(
        {
            "n_steps": n_steps,
            "phase_name": phase_name,
            "phase_idx": phase_idx,
            "load_from": load_from or "none",
        }
    )

    # Training loop
    logger.info(f"Starting training for {n_steps} steps...")
    for _ in range(n_steps):
        runner.learning_step()

    # Save checkpoint (includes baseline for curriculum continuity)
    checkpoint_dir = Path(log_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "training_state.pt")
    runner.dump_training_state(checkpoint_path)
    logger.info(f"Saved training state to: {checkpoint_path}")
    # Also save player-only checkpoint for evaluation/inference
    player_path = str(checkpoint_dir / "player.pt")
    runner.dump_player(player_path)
    logger.info(f"Saved player to: {player_path}")

    # Cleanup
    runner.close_tensorboard()

    # Return training state path for curriculum continuity
    return checkpoint_path


def main(
    config_path: Annotated[str, typer.Argument(help="Path to config file (YAML or JSON)")],
    phase: Annotated[
        int | None,
        typer.Option(help="Run only specific phase (0-indexed). If not set, runs all phases."),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(help="Resume training from the last checkpoint (same TensorBoard run)"),
    ] = False,
    log_to_file: Annotated[
        bool, typer.Option(help="Also log to file in experiment folder")
    ] = False,
    log_dir: Annotated[str, typer.Option(help="Directory for logs and checkpoints")] = "runs",
) -> None:
    """Run training from a configuration file.

    Use --resume to continue training from an existing checkpoint,
    preserving the TensorBoard timeline.
    """
    # Setup logging - always log to stdout
    logger.remove()
    logger.add(sys.stdout)

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")

    # Get experiment name
    experiment_name = config.get("experiment_name", Path(config_path).stem)
    logger.info(f"Experiment: {experiment_name}")

    # Add file logging to experiment folder if requested
    if log_to_file:
        experiment_dir = Path(log_dir) / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = experiment_dir / "experiment.log"
        logger.add(log_file_path)
        logger.info(f"Logging to: {log_file_path}")

    # Get phases
    phases = config.get("phases", [])
    if not phases:
        # Single-phase config (backward compatibility)
        phases = [config]

    logger.info(f"Found {len(phases)} training phase(s)")

    # Handle --resume: find the latest checkpoint and continue from there
    if resume:
        checkpoint_path = None
        has_phases_section = "phases" in config and config["phases"]

        if has_phases_section:
            # Config has phases section: look in phase subdirectories
            for i in range(len(phases) - 1, -1, -1):
                phase_name = phases[i].get("name", f"phase_{i}")
                candidate = Path(log_dir) / experiment_name / phase_name / "training_state.pt"
                if candidate.exists():
                    checkpoint_path = candidate
                    phase = i  # Resume from this phase
                    break
                candidate = Path(log_dir) / experiment_name / phase_name / "player.pt"
                if candidate.exists():
                    checkpoint_path = candidate
                    phase = i
                    break
        else:
            # Legacy single-phase config (no phases section): checkpoint in experiment root
            candidate = Path(log_dir) / experiment_name / "training_state.pt"
            if candidate.exists():
                checkpoint_path = candidate
            else:
                candidate = Path(log_dir) / experiment_name / "player.pt"
                if candidate.exists():
                    checkpoint_path = candidate

        if checkpoint_path is None or not checkpoint_path.exists():
            experiment_dir = Path(log_dir) / experiment_name
            raise FileNotFoundError(
                f"No checkpoint found to resume from in {experiment_dir}. "
                "Run without --resume to start fresh."
            )

        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

    # Determine which phases to run
    resume_checkpoint: str | None = str(checkpoint_path) if resume else None

    if phase is not None:
        if phase < 0 or phase >= len(phases):
            raise ValueError(f"Phase {phase} out of range (0-{len(phases)-1})")
        phases_to_run = [(phase, phases[phase])]
        # Try to find previous checkpoint (prefer training_state.pt, fall back to player.pt)
        if phase > 0 and not resume:
            prev_phase_name = phases[phase - 1].get("name", f"phase_{phase - 1}")
            prev_training_state = (
                Path(log_dir) / experiment_name / prev_phase_name / "training_state.pt"
            )
            prev_player = Path(log_dir) / experiment_name / prev_phase_name / "player.pt"
            if prev_training_state.exists():
                previous_checkpoint = str(prev_training_state)
            elif prev_player.exists():
                previous_checkpoint = str(prev_player)
            else:
                previous_checkpoint = None
        else:
            previous_checkpoint = None
    else:
        phases_to_run = list(enumerate(phases))
        previous_checkpoint = None

    global_config = config.get("defaults", {})

    for i, (phase_idx, phase_config) in enumerate(phases_to_run):
        resolved_phase_config = resolve_phase_config(config, global_config, phase_config)
        current_resume_from = resume_checkpoint if i == 0 else None

        checkpoint = run_phase(
            phase_config=resolved_phase_config,
            phase_idx=phase_idx,
            base_experiment_name=experiment_name,
            log_dir=log_dir,
            previous_checkpoint=previous_checkpoint,
            resume_from=current_resume_from,
        )
        previous_checkpoint = checkpoint

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Logs saved to: {log_dir}/{experiment_name}")
    logger.info("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

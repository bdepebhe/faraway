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

from faraway.torch.learning_runner import LearningRunner
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.transformers_player import TransformersPlayer


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


def get_default_model_params(player_type: str) -> dict[str, Any]:
    """Get default model parameters for a player type."""
    if player_type == "mlp":
        return {
            "hidden_layers_sizes": [512, 512],
            "dropout_rate": 0.1,
        }
    elif player_type == "transformer":
        return {
            "embed_dim": 64,
            "n_attention_heads": 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def get_default_player_params(player_type: str) -> dict[str, Any]:
    """Get default player parameters for a player type."""
    if player_type == "mlp":
        return {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
    elif player_type == "transformer":
        return {
            "use_mode_embedding": False,
        }
    else:
        raise ValueError(f"Unknown player type: {player_type}")


def run_phase(
    phase_config: dict[str, Any],
    global_config: dict[str, Any],
    root_config: dict[str, Any],
    phase_idx: int,
    base_experiment_name: str,
    log_dir: str,
    previous_checkpoint: str | None = None,
    resume_from: str | None = None,
) -> str:
    """Run a single training phase.

    Args:
        phase_config: Configuration for this phase
        global_config: Global configuration (from 'defaults' section)
        root_config: Root configuration (top-level keys)
        phase_idx: Phase index (0-based)
        base_experiment_name: Base name for the experiment
        log_dir: Directory for logs and checkpoints
        previous_checkpoint: Path to checkpoint from previous phase (optional)
        resume_from: Path to checkpoint to resume from (overrides load_from, for --resume)

    Returns:
        Path to the saved checkpoint from this phase
    """
    phase_name = phase_config.get("name", f"phase_{phase_idx}")
    experiment_name = f"{base_experiment_name}/{phase_name}"

    logger.info("=" * 60)
    logger.info(f"Starting Phase {phase_idx}: {phase_name}")
    logger.info("=" * 60)

    # Helper to get config value: phase -> global defaults -> root config -> default
    def get_config(key: str, default: Any = None) -> Any:
        if key in phase_config:
            return phase_config[key]
        if key in global_config:
            return global_config[key]
        if key in root_config:
            return root_config[key]
        return default

    # Merge global defaults with phase-specific config
    player_type = get_config("player_type", "transformer")

    # Model params: defaults -> root config -> global defaults -> phase config
    model_params = get_default_model_params(player_type)
    model_params.update(root_config.get("model_params", {}))
    model_params.update(global_config.get("model_params", {}))
    model_params.update(phase_config.get("model_params", {}))

    # Player params: same merge strategy
    player_params = get_default_player_params(player_type)
    player_params.update(root_config.get("player_params", {}))
    player_params.update(global_config.get("player_params", {}))
    player_params.update(phase_config.get("player_params", {}))

    # RL params
    # Note: when peer_relative_reward=True, prior_baseline_score and update_baseline_rate
    # are NOT used for training (advantage from batch mean), but baseline EMA is still tracked.
    rl_params = {
        "prior_baseline_score": 29,  # not used for training if peer_relative_reward=True
        "train_batch_size": 32,
        "update_baseline_rate": 0.05,  # not used for training if peer_relative_reward=True
        "peer_relative_reward": False,
        "advantage_peer_normalization": "zscore",  # "zscore" or "center" (only if
        # peer_relative_reward=True)
        "peer_sub_batch_size": None,  # sub-batch size for peer comparison (None = full batch)
    }
    rl_params.update(root_config.get("rl_params", {}))
    rl_params.update(global_config.get("rl_params", {}))
    rl_params.update(phase_config.get("rl_params", {}))

    # Optimizer params
    optimizer_params = {"lr": 0.0005}
    optimizer_params.update(root_config.get("optimizer_params", {}))
    optimizer_params.update(global_config.get("optimizer_params", {}))
    optimizer_params.update(phase_config.get("optimizer_params", {}))

    # Game params
    n_rounds = get_config("n_rounds", 8)
    draft_size = get_config("draft_size", 10)
    use_hand = get_config("use_hand", False)
    n_cards_hand = get_config("n_cards_hand", 3)
    use_bonus_cards = get_config("use_bonus_cards", True)

    # Training params
    n_steps = phase_config.get("n_steps", root_config.get("n_steps", 1000))
    verbose = get_config("verbose", 1)

    # Evaluation configs (support both naming conventions, check all config levels)
    eval_vs_random_config = None
    for key in ["eval_vs_random_config", "eval_vs_random"]:
        if key in phase_config or key in global_config or key in root_config:
            eval_vs_random_config = root_config.get(key, {}).copy()
            eval_vs_random_config.update(global_config.get(key, {}))
            eval_vs_random_config.update(phase_config.get(key, {}))
            break

    eval_solo_config = None
    for key in ["eval_solo_config", "eval_solo"]:
        if key in phase_config or key in global_config or key in root_config:
            eval_solo_config = root_config.get(key, {}).copy()
            eval_solo_config.update(global_config.get(key, {}))
            eval_solo_config.update(phase_config.get(key, {}))
            break

    # Checkpoint loading
    # Priority: resume_from (--resume flag) > phase_config > previous_checkpoint
    if resume_from:
        load_from = resume_from
        logger.info("Resuming training from checkpoint (--resume mode)")
    else:
        load_from = phase_config.get("load_from", None)
        if load_from == "previous" and previous_checkpoint:
            load_from = previous_checkpoint
        elif load_from == "previous":
            load_from = None  # No previous checkpoint available

    # Log configuration
    logger.info(f"Player type: {player_type}")
    logger.info(f"Model params: {model_params}")
    logger.info(f"Player params: {player_params}")
    logger.info(f"RL params: {rl_params}")
    logger.info(f"Optimizer params: {optimizer_params}")
    logger.info(f"Use draft: {use_hand}")
    logger.info(f"N steps: {n_steps}")
    if rl_params.get("peer_relative_reward"):
        sub_batch_size = rl_params.get("peer_sub_batch_size")
        sub_batch_info = f", sub_batch_size={sub_batch_size}" if sub_batch_size else " (full batch)"
        logger.info(
            f"Peer-relative reward: enabled"
            f"(normalization={rl_params.get('advantage_peer_normalization', 'zscore')}"
            f"{sub_batch_info})"
        )
    if load_from:
        logger.info(f"Loading from: {load_from}")

    # Create the runner (default setting = solo)
    game = LearningRunner(
        n_rounds=n_rounds,
        draft_size=draft_size,
        use_hand=use_hand,
        n_cards_hand=n_cards_hand,
        use_bonus_cards=use_bonus_cards,
        model_params=model_params,
        player_params=player_params,
        player_type=player_type,
        optimizer_params=optimizer_params,
        rl_params=rl_params,
        verbose=verbose,
        experiment_name=experiment_name,
        log_dir=log_dir,
        eval_vs_random_config=eval_vs_random_config,
        eval_solo_config=eval_solo_config,
    )

    # Log trainable parameters count
    n_trainable_params = sum(
        p.numel() for p in game.players[0].model.parameters() if p.requires_grad
    )
    logger.info(f"Trainable parameters: {n_trainable_params:,}")

    # Load checkpoint if specified
    if load_from:
        logger.info(f"Loading training state from: {load_from}")
        # Try loading as training state first (includes baseline)
        try:
            game.load_training_state(load_from)
            logger.info(
                f"Loaded training state: {game.players[0].n_training_games_played} games, "
                f"baseline={game.baseline:.2f}"
            )
        except KeyError:
            # Fall back to loading as player checkpoint (no baseline)
            logger.info("Falling back to player-only checkpoint (no baseline)")
            loaded_player: MLPPlayer | TransformersPlayer
            if player_type == "mlp":
                loaded_player = MLPPlayer.load(load_from, device=game.device)
            elif player_type == "transformer":
                loaded_player = TransformersPlayer.load(load_from, device=game.device)
            else:
                raise ValueError(f"Unknown player type: {player_type}") from None
            game.players[0].model.load_state_dict(loaded_player.model.state_dict())
            game.players[0].n_training_games_played = loaded_player.n_training_games_played
            logger.info(f"Loaded player with {loaded_player.n_training_games_played} games played")

    # Handle initial_baseline setting
    # Options:
    #   - "previous": keep from checkpoint
    #   - "from_eval": run solo eval and use mean score as baseline
    #   - number: set specific value
    #   - not set: use rl_params default
    initial_baseline = phase_config.get("initial_baseline", None)
    if initial_baseline is not None:
        if initial_baseline == "previous":
            # Keep the baseline from checkpoint (already loaded above)
            logger.info(f"Keeping baseline from checkpoint: {game.baseline:.2f}")
        elif initial_baseline == "from_eval":
            # Run solo evaluation and use mean score as baseline
            game.initialize_baseline_from_eval()
        elif isinstance(initial_baseline, int | float):
            game.baseline = float(initial_baseline)
            logger.info(f"Set initial baseline to: {game.baseline:.2f}")
        else:
            raise ValueError(
                f"Invalid initial_baseline value: {initial_baseline}. "
                "Expected 'previous', 'from_eval', or a number."
            )
    elif not load_from:
        # No checkpoint loaded and no initial_baseline specified - use rl_params default
        logger.info(f"Using default baseline: {game.baseline:.2f}")

    # Log hyperparameters
    game.log_hparams(
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
        game.learning_step()

    # Save checkpoint (includes baseline for curriculum continuity)
    checkpoint_dir = Path(log_dir) / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(checkpoint_dir / "training_state.pt")
    game.dump_training_state(checkpoint_path)
    logger.info(f"Saved training state to: {checkpoint_path}")
    # Also save player-only checkpoint for evaluation/inference
    player_path = str(checkpoint_dir / "player.pt")
    game.dump_player(player_path)
    logger.info(f"Saved player to: {player_path}")

    # Cleanup
    game.close_tensorboard()

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

    # Run phases
    for i, (phase_idx, phase_config) in enumerate(phases_to_run):
        # Only use resume_from for the first phase when resuming
        current_resume_from = resume_checkpoint if i == 0 else None

        checkpoint = run_phase(
            phase_config=phase_config,
            global_config=config.get("defaults", {}),
            root_config=config,
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

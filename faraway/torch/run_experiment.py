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

from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.solo_learning import SoloLearningGame
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
            "use_cards_hand_in_state": False,
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
    rl_params = {
        "prior_baseline_score": 29,
        "train_batch_size": 32,
        "update_baseline_rate": 0.05,
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
    use_draft = get_config("use_draft", False)
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
    logger.info(f"Use draft: {use_draft}")
    logger.info(f"N steps: {n_steps}")
    if load_from:
        logger.info(f"Loading from: {load_from}")

    # Create the game
    game = SoloLearningGame(
        n_rounds=n_rounds,
        draft_size=draft_size,
        use_draft=use_draft,
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
    # Options: "previous" (keep from checkpoint), number (set specific value),
    # or not set (use rl_params default)
    initial_baseline = phase_config.get("initial_baseline", None)
    if initial_baseline is not None:
        if initial_baseline == "previous":
            # Keep the baseline from checkpoint (already loaded above)
            logger.info(f"Keeping baseline from checkpoint: {game.baseline:.2f}")
        elif isinstance(initial_baseline, int | float):
            game.baseline = float(initial_baseline)
            logger.info(f"Set initial baseline to: {game.baseline:.2f}")
        else:
            raise ValueError(f"Invalid initial_baseline value: {initial_baseline}")
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
    log_to_file: Annotated[bool, typer.Option(help="Log to file instead of stdout")] = False,
    log_dir: Annotated[str, typer.Option(help="Directory for logs and checkpoints")] = "runs",
) -> None:
    """Run training from a configuration file."""
    # Setup logging
    logger.remove()
    if log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    # Load configuration
    config = load_config(config_path)
    logger.info(f"Loaded config from: {config_path}")

    # Get experiment name
    experiment_name = config.get("experiment_name", Path(config_path).stem)
    logger.info(f"Experiment: {experiment_name}")

    # Get phases
    phases = config.get("phases", [])
    if not phases:
        # Single-phase config (backward compatibility)
        phases = [config]

    logger.info(f"Found {len(phases)} training phase(s)")

    # Determine which phases to run
    if phase is not None:
        if phase < 0 or phase >= len(phases):
            raise ValueError(f"Phase {phase} out of range (0-{len(phases)-1})")
        phases_to_run = [(phase, phases[phase])]
        # Try to find previous checkpoint (prefer training_state.pt, fall back to player.pt)
        if phase > 0:
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
    for phase_idx, phase_config in phases_to_run:
        checkpoint = run_phase(
            phase_config=phase_config,
            global_config=config.get("defaults", {}),
            root_config=config,
            phase_idx=phase_idx,
            base_experiment_name=experiment_name,
            log_dir=log_dir,
            previous_checkpoint=previous_checkpoint,
        )
        previous_checkpoint = checkpoint

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Logs saved to: {log_dir}/{experiment_name}")
    logger.info("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

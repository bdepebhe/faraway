"""
Resolve experiment config into block-wise rl_params and full phase config.

- resolve_rl_params: merge rl_params (nested) from root / global / phase.
- resolve_phase_config: merge everything for one phase into a single dict suitable
  for run_phase (includes LearningRunner kwargs plus phase-only keys: n_steps, name,
  load_from, initial_baseline).
"""

from copy import deepcopy
from typing import Any


def get_default_model_params(player_type: str) -> dict[str, Any]:
    if player_type == "mlp":
        return {"hidden_layers_sizes": [512, 512], "dropout_rate": 0.1}
    if player_type == "transformer":
        return {
            "embed_dim": 64,
            "n_attention_heads": 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }
    raise ValueError(f"Unknown player type: {player_type}")


def get_default_player_params(player_type: str) -> dict[str, Any]:
    if player_type == "mlp":
        return {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
    if player_type == "transformer":
        return {"use_mode_embedding": False}
    raise ValueError(f"Unknown player type: {player_type}")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge override into base recursively. override wins; base is not mutated."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


# Default rl_params with sub-blocks for **block_params usage
DEFAULT_RL_PARAMS: dict[str, Any] = {
    "train_batch_size": 32,
    "advantage": {
        "prior_baseline_score": 29,
        "update_baseline_rate": 0.05,
        "peer_relative_reward": False,
        "advantage_peer_normalization": "zscore",
        "peer_sub_batch_size": None,
    },
    "algorithm": {
        "grad_clip": None,
    },
    "temperature": {
        "initial": 1.0,
        "decay": 1.0,
    },
    "optimizer_params": {"lr": 0.0005},
}


def resolve_rl_params(
    root_config: dict[str, Any],
    global_config: dict[str, Any],
    phase_config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve rl_params (nested) from config. Merge order: DEFAULT_RL_PARAMS
    <- root <- global <- phase."""
    raw: dict[str, Any] = {}
    for config in (root_config, global_config, phase_config):
        if "rl_params" in config:
            raw = _deep_merge(raw, config["rl_params"])
    return _deep_merge(DEFAULT_RL_PARAMS, raw)


def _get(
    key: str, default: Any, root: dict[str, Any], global_cfg: dict[str, Any], phase: dict[str, Any]
) -> Any:
    if key in phase:
        return phase[key]
    if key in global_cfg:
        return global_cfg[key]
    if key in root:
        return root[key]
    return default


def resolve_phase_config(
    root_config: dict[str, Any],
    global_config: dict[str, Any],
    phase_config: dict[str, Any],
) -> dict[str, Any]:
    """Merge root / global / phase into one resolved phase config.

    Result can be passed to run_phase and used as LearningRunner(**runner_kwargs)
    (after adding experiment_name, log_dir). Contains phase-only keys: n_steps,
    name, load_from, initial_baseline.
    """
    root, global_cfg, phase = root_config, global_config, phase_config
    player_type = _get("player_type", "transformer", root, global_cfg, phase)

    model_params = get_default_model_params(player_type)
    model_params.update(root.get("model_params", {}))
    model_params.update(global_cfg.get("model_params", {}))
    model_params.update(phase.get("model_params", {}))

    player_params = get_default_player_params(player_type)
    player_params.update(root.get("player_params", {}))
    player_params.update(global_cfg.get("player_params", {}))
    player_params.update(phase.get("player_params", {}))

    rl_params = resolve_rl_params(root, global_cfg, phase)
    optimizer_overlay = {"lr": 0.0005}
    optimizer_overlay.update(root.get("optimizer_params", {}))
    optimizer_overlay.update(global_cfg.get("optimizer_params", {}))
    optimizer_overlay.update(phase.get("optimizer_params", {}))
    rl_params.setdefault("optimizer_params", {}).update(optimizer_overlay)

    eval_flows = (
        phase.get("eval_flows") or global_cfg.get("eval_flows") or root.get("eval_flows") or []
    )
    eval_flows = list(eval_flows)

    return {
        "n_rounds": _get("n_rounds", 8, root, global_cfg, phase),
        "draft_size": _get("draft_size", 10, root, global_cfg, phase),
        "replace_remaining_cards": _get("replace_remaining_cards", True, root, global_cfg, phase),
        "use_bonus_cards": _get("use_bonus_cards", True, root, global_cfg, phase),
        "use_hand": _get("use_hand", False, root, global_cfg, phase),
        "n_cards_hand": _get("n_cards_hand", 3, root, global_cfg, phase),
        "verbose": _get("verbose", 1, root, global_cfg, phase),
        "model_params": model_params,
        "player_params": player_params,
        "player_type": player_type,
        "rl_params": rl_params,
        "eval_flows": eval_flows if eval_flows else None,
        "n_steps": phase.get("n_steps", root.get("n_steps", 1000)),
        "name": phase.get("name", "phase"),
        "load_from": phase.get("load_from"),
        "initial_baseline": phase.get("initial_baseline"),
    }

"""Solo evaluation of a trained player: load player, run games in env, report scores.

Uses only the player and BatchedFarawayEnv—no LearningRunner (no training stack).
"""

import argparse

import torch
from loguru import logger

from faraway.torch.base_game import IDX_ID
from faraway.torch.env import BatchedFarawayEnv
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.transformers_player import TransformersPlayer

# Temperature for evaluation (near-greedy; 0 would make softmax undefined)
EVAL_TEMPERATURE = 1e-8


def load_player(path: str, device: torch.device | None = None) -> MLPPlayer | TransformersPlayer:
    """Load player from checkpoint. Detects type from checkpoint or path."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    player_type = checkpoint.get("player_type", "mlp")
    if player_type == "mlp":
        return MLPPlayer.load(path, device=device)
    if player_type == "transformer":
        return TransformersPlayer.load(path, device=device)
    raise ValueError(f"Unknown player_type in checkpoint: {player_type}")


def run_solo_eval(
    player: MLPPlayer | TransformersPlayer,
    n_batches: int,
    batch_size: int,
    draft_size: int = 10,
    use_hand: bool = False,
    n_cards_hand: int = 3,
    temperature: float = EVAL_TEMPERATURE,
    collect_best_games: int = 0,
) -> tuple[torch.Tensor, list[tuple[float, torch.Tensor, torch.Tensor]]]:
    """Run solo games with the player in BatchedFarawayEnv.

    Returns:
        scores: (n_batches * batch_size,) tensor.
        best_games: If collect_best_games > 0, list of (score, main_field, bonus_field) for
            every game (caller can sort and take top N).
    """
    env = BatchedFarawayEnv(
        n_rounds=player.n_rounds,
        use_bonus_cards=player.use_bonus_cards,
        draft_size=draft_size,
        use_hand=use_hand,
        n_cards_hand=n_cards_hand,
        device=player.device,
    )
    all_scores = []
    best_games: list[tuple[float, torch.Tensor, torch.Tensor]] = []

    for _ in range(n_batches):
        env.reset(batch_size, [player])
        for _ in range(env.n_rounds):
            env.step_round([player], temperature=temperature)
        scores = env.get_scores([player])[:, 0]  # (batch,)
        all_scores.append(scores)
        if collect_best_games > 0:
            for i in range(batch_size):
                best_games.append(
                    (
                        scores[i].item(),
                        player.fields["main"][i].clone(),
                        player.fields["bonus"][i].clone(),
                    )
                )

    return torch.cat(all_scores), best_games


def main() -> None:
    parser = argparse.ArgumentParser(description="Solo evaluation of a trained player")
    parser.add_argument("player_path", type=str, help="Path to the player checkpoint (.pt)")
    parser.add_argument(
        "--draft-size", type=int, default=10, help="Number of cards in draft (default: 10)"
    )
    parser.add_argument(
        "--n-batches", type=int, default=100, help="Number of batches to run (default: 100)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--use-hand", action="store_true", help="Use hand mechanism (draft from hand)"
    )
    parser.add_argument(
        "--n-cards-hand", type=int, default=3, help="Number of cards in hand (default: 3)"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level (default: 1)")
    parser.add_argument(
        "--show-best", type=int, default=0, help="Show details of N best games (default: 0)"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    player = load_player(args.player_path, device=device)

    logger.info(f"Loaded player from {args.player_path}")
    logger.info(
        f"Draft size: {args.draft_size}, Batch size: {args.batch_size}, N batches: {args.n_batches}"
    )

    scores, best_games = run_solo_eval(
        player,
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        draft_size=args.draft_size,
        use_hand=args.use_hand,
        n_cards_hand=args.n_cards_hand,
        collect_best_games=args.show_best,
    )

    total_games = len(scores)
    mean_score = scores.mean().item()
    std_score = scores.std().item()
    min_score = scores.min().item()
    max_score = scores.max().item()

    logger.info("=" * 50)
    logger.info(f"Results ({total_games} games):")
    logger.info(f"  Mean score: {mean_score:.2f} ± {std_score:.2f}")
    logger.info(f"  Min score:  {min_score:.0f}")
    logger.info(f"  Max score:  {max_score:.0f}")
    logger.info("=" * 50)

    if args.show_best > 0 and best_games:
        best_games.sort(key=lambda x: x[0], reverse=True)
        logger.info(f"\nTop {args.show_best} games:")
        for rank, (score, main_field, bonus_field) in enumerate(best_games[: args.show_best], 1):
            main_card_ids = main_field[:, IDX_ID].long().tolist()
            bonus_card_ids = [
                int(bonus_field[j, IDX_ID].item())
                for j in range(bonus_field.shape[0])
                if bonus_field[j, IDX_ID].item() > 0
            ]
            logger.info(f"\n  #{rank}: Score = {score:.0f}")
            logger.info(f"    Main cards (IDs): {main_card_ids}")
            logger.info(f"    Bonus cards (IDs): {bonus_card_ids}")


if __name__ == "__main__":
    main()

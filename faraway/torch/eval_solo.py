"""Simple solo evaluation script for trained players."""

import argparse
from typing import Any

import torch
from loguru import logger

from faraway.torch.solo_learning import SoloLearningGame


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

    # Load checkpoint to detect player type
    checkpoint = torch.load(args.player_path, map_location="cpu", weights_only=False)
    player_type = checkpoint.get("player_type", "mlp")

    logger.info(f"Loading {player_type} player from {args.player_path}")
    logger.info(
        f"Draft size: {args.draft_size}, Batch size: {args.batch_size}, N batches: {args.n_batches}"
    )

    # Extract player config from checkpoint
    player_config = checkpoint.get("config", {})
    n_rounds = player_config.get("n_rounds", 8)
    use_bonus_cards = player_config.get("use_bonus_cards", True)

    # Build player params from checkpoint config
    player_params: dict[str, Any] = {}
    if player_type == "transformer":
        player_params["use_mode_embedding"] = player_config.get("use_mode_embedding", False)

    model_params = checkpoint.get("model_params", {})

    # Create game with the loaded configuration
    game = SoloLearningGame(
        n_rounds=n_rounds,
        draft_size=args.draft_size,
        use_bonus_cards=use_bonus_cards,
        use_hand=args.use_hand,
        n_cards_hand=args.n_cards_hand,
        verbose=args.verbose,
        player_type=player_type,
        player_params=player_params,
        model_params=model_params,
        model_path=args.player_path,  # Load the trained weights
    )

    # Run evaluation
    logger.info(f"Running {args.n_batches} batches of {args.batch_size} games...")

    all_scores = []
    best_games = []  # Store (score, main_field, bonus_field)

    for _ in range(args.n_batches):
        game.reset_games_batch(batch_size=args.batch_size)
        game.deal_initial_hands()

        for _ in range(n_rounds):
            game.play_round()

        scores = game.get_scores()
        all_scores.append(scores)

        # Track best games if requested
        if args.show_best > 0:
            for i in range(args.batch_size):
                score = scores[i].item()
                best_games.append(
                    (
                        score,
                        game.players[0].fields["main"][i].clone(),
                        game.players[0].fields["bonus"][i].clone(),
                    )
                )

    # Aggregate results
    all_scores_tensor = torch.cat(all_scores)
    mean_score = all_scores_tensor.mean().item()
    std_score = all_scores_tensor.std().item()
    min_score = all_scores_tensor.min().item()
    max_score = all_scores_tensor.max().item()

    total_games = args.n_batches * args.batch_size

    logger.info("=" * 50)
    logger.info(f"Results ({total_games} games):")
    logger.info(f"  Mean score: {mean_score:.2f} ± {std_score:.2f}")
    logger.info(f"  Min score:  {min_score:.0f}")
    logger.info(f"  Max score:  {max_score:.0f}")
    logger.info("=" * 50)

    # Show best games if requested
    if args.show_best > 0:
        from faraway.torch.base_game import IDX_ID

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

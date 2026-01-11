import sys
from typing import Annotated

import typer
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.nn_player import BaseNNPlayer
from faraway.torch.random_player import RandomPlayer
from faraway.torch.real_game import RealNNGame


def play_vs_random(
    player: BaseNNPlayer,
    n_random_players: int,
    n_eval_batches: int,
    batch_size: int,
    experiment_name: str | None = None,
    log_dir: str = "runs",
    writer: SummaryWriter | None = None,
    verbose: int = 2,
) -> tuple[float, float]:
    """Play a game against random players and log results.

    Args:
        player: The player to evaluate
        n_random_players: Number of random opponents
        n_eval_batches: Number of evaluation batches
        batch_size: Games per batch
        experiment_name: TensorBoard experiment name (required if writer is None)
        log_dir: Directory for TensorBoard logs
        writer: Optional existing TensorBoard writer. If provided, won't create/close one.
        verbose: Verbosity level

    Returns:
        Tuple of (win_rate, mean_score)
    """
    random_players: list[BaseNNPlayer] = [RandomPlayer() for _ in range(n_random_players)]
    all_players: list[BaseNNPlayer] = [player] + random_players
    game = RealNNGame(
        players=all_players,
        n_rounds=8,
        verbose=verbose,
        experiment_name=experiment_name,
        log_dir=log_dir,
    )

    # Use provided writer or create a new one
    owns_writer = writer is None
    if owns_writer:
        game.init_tensorboard()
    else:
        game.writer = writer

    wins, mean_scores = game.run_tournament(n_batches=n_eval_batches, batch_size=batch_size)
    step = player.n_training_games_played

    win_rate = wins[0] / (n_eval_batches * batch_size)
    mean_score = mean_scores[0]

    if game.writer is not None:
        game.writer.add_scalar(f"eval/vs_{n_random_players}_random/win_rate", win_rate, step)
        game.writer.add_scalar(f"eval/vs_{n_random_players}_random/mean_score", mean_score, step)
        game.writer.flush()

    # Only close if we created the writer
    if owns_writer:
        game.close_tensorboard()

    return win_rate, mean_score


def main(
    path_to_player: Annotated[str, typer.Argument(help="Path to the player to use")],
    log_to_file: Annotated[bool, typer.Option(help="Whether to log to a file")] = False,
    experiment_name: Annotated[
        str | None, typer.Option(help="Name for TensorBoard experiment")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 32,
    n_eval_batches: Annotated[int, typer.Option(help="Number of evaluation batches")] = 100,
    n_random_players: Annotated[int, typer.Option(help="Number of random players")] = 1,
) -> None:
    """Run a game against a random player."""
    logger.remove()  # remove default stderr handler
    if log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    player = MLPPlayer.load(path_to_player)

    win_rate, mean_score = play_vs_random(
        player, n_random_players, n_eval_batches, batch_size, experiment_name
    )
    logger.info(f"Win rate: {win_rate:.2%}, Mean score: {mean_score:.2f}")


if __name__ == "__main__":
    typer.run(main)

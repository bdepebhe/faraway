"""Base class for batched NN games."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import BonusCard, MainCard
from faraway.core.final_count import final_count
from faraway.core.player_field import PlayerField
from faraway.torch.load_cards import load_bonus_deck_tensor, load_main_deck_tensor

# =============================================================================
# Lazy-loaded global deck tensors
# =============================================================================

_MAIN_DECK_TENSOR: torch.Tensor | None = None
_BONUS_DECK_TENSOR: torch.Tensor | None = None


def get_main_deck_tensor() -> torch.Tensor:
    """Get the global main deck tensor (lazy loaded)."""
    global _MAIN_DECK_TENSOR
    if _MAIN_DECK_TENSOR is None:
        _MAIN_DECK_TENSOR = load_main_deck_tensor()
    return _MAIN_DECK_TENSOR


def get_bonus_deck_tensor() -> torch.Tensor:
    """Get the global bonus deck tensor (lazy loaded)."""
    global _BONUS_DECK_TENSOR
    if _BONUS_DECK_TENSOR is None:
        _BONUS_DECK_TENSOR = load_bonus_deck_tensor()
    return _BONUS_DECK_TENSOR


def final_count_from_tensor_field(
    main_field_tensor: torch.Tensor, bonus_field_tensor: torch.Tensor
) -> torch.Tensor:
    main_cards = [
        MainCard.from_numpy(main_field_tensor[i, :]) for i in range(main_field_tensor.shape[0])
    ]
    bonus_cards = [
        BonusCard.from_main_card(MainCard.from_numpy(bonus_field_tensor[i, :]))
        for i in range(bonus_field_tensor.shape[0])
    ]
    return final_count(
        PlayerField(
            main_cards=main_cards, bonus_cards=bonus_cards, n_rounds=main_field_tensor.shape[0]
        )
    )


class BaseNNGame(ABC):
    """
    Batched game using tensors.

    Attributes:
        n_rounds: Number of rounds in the game
        players: List of players in the game
        use_bonus_cards: Whether to use bonus cards
        device: Torch device for tensors
        verbose: Verbosity level

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
        use_bonus_cards: bool = True,
        device: torch.device | None = None,
        players: Sequence[BasePlayer] | None = None,
        verbose: int = 0,
        experiment_name: str | None = None,
        log_dir: str = "runs",
    ):
        self.n_rounds = n_rounds
        self.use_bonus_cards = use_bonus_cards
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_deck = get_main_deck_tensor()
        bonus_deck = get_bonus_deck_tensor()
        self.decks = {
            "main": main_deck,
            "bonus": bonus_deck,
        }
        self.picked_probabilities: torch.Tensor  # (used for training only)

        self.players: Sequence[BasePlayer] = players or []
        # check that the players have the same number of rounds
        self.verbose = verbose

        # TensorBoard logging (optional)
        self.writer: SummaryWriter | None = None
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.total_games_played = 0

    def init_tensorboard(
        self, experiment_name: str | None = None, default_prefix: str = "faraway"
    ) -> None:
        """Initialize TensorBoard writer for logging.

        Args:
            experiment_name: Name for the experiment. If None/empty, uses self.experiment_name
                or generates a timestamped name.
            default_prefix: Prefix for auto-generated experiment names.
        """
        if not experiment_name:
            experiment_name = self.experiment_name
        if not experiment_name:
            experiment_name = f"{default_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=f"{self.log_dir}/{experiment_name}")

    def close_tensorboard(self) -> None:
        """Close the TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    @abstractmethod
    def play_round(self) -> torch.Tensor:
        pass

    def reset_games_batch(self, batch_size: int) -> None:
        main_deck_availability = torch.ones(
            batch_size, self.decks["main"].shape[0], dtype=torch.bool, device=self.device
        )  # (batch, n_total_main_cards)
        bonus_deck_availability = torch.ones(
            batch_size, self.decks["bonus"].shape[0], dtype=torch.bool, device=self.device
        )  # (batch, n_total_bonus_cards)
        self.deck_availability = {
            "main": main_deck_availability,
            "bonus": bonus_deck_availability,
        }
        for player in self.players:
            player.reset_games_batch(batch_size)
        self.round_index = 0

    def get_used_cards_ids(self, type: str, game_id: int) -> list[int]:
        used_cards_ids = [
            1 + card_id
            for card_id in torch.where(~self.deck_availability[type][game_id])[0].tolist()
        ]
        return used_cards_ids

    def get_scores(self) -> torch.Tensor:
        return torch.stack(
            [
                torch.tensor(
                    [
                        final_count_from_tensor_field(
                            player.fields["main"][i, :, :], player.fields["bonus"][i, :, :]
                        )
                        for i in range(player.fields["main"].shape[0])
                    ],
                    dtype=torch.float32,
                    device=self.device,
                )
                for player in self.players
            ],
            dim=1,
        )  # (batch, players)

    def play_games_batch(self, batch_size: int, learning_mode: bool = False) -> None:
        self.reset_games_batch(batch_size)
        for _ in range(self.n_rounds):
            picked_probabilities = self.play_round()
            if learning_mode:
                # add the probabilities to the picked_probabilities tensor
                self.picked_probabilities = torch.concat(
                    [self.picked_probabilities, picked_probabilities], dim=1
                )

    def play_games_batches(self, n_batches: int, batch_size: int) -> torch.Tensor:
        if self.verbose > 0:
            logger.info(f"Playing {n_batches} batches of {batch_size} games")
        batches_scores: list[torch.Tensor] = []
        for i in range(n_batches):
            self.play_games_batch(batch_size)
            if self.verbose > 1:
                logger.info(f"Batch {i + 1} completed")
            batches_scores.append(self.get_scores())
        scores = torch.cat(batches_scores, dim=0)  # (n_batches * batch_size, players)
        return scores

    def run_tournament(
        self,
        n_batches: int,
        batch_size: int,
        player_names: list[str] | None = None,
    ) -> tuple[list[int], list[float]]:
        """Run a tournament and return wins and mean scores per player.

        Args:
            n_batches: Number of batches to play
            batch_size: Number of games per batch
            player_names: Optional names for TensorBoard logging

        Returns:
            Tuple of (wins per player, mean scores per player)
        """
        scores = self.play_games_batches(n_batches, batch_size)
        self.total_games_played += n_batches * batch_size

        winner = scores.argmax(dim=1)
        wins = []
        win_rate = []
        for player_id in range(len(self.players)):
            wins.append(torch.where(winner == player_id)[0].shape[0])
            win_rate.append(wins[-1] / (n_batches * batch_size) * 100)

        mean_scores = scores.mean(dim=0).tolist()
        if self.verbose > 0:
            logger.info(
                f"Tournament completed.\n"
                f"Mean scores: {mean_scores}\n"
                f"Wins: {wins}\n"
                f"Win rate: {win_rate}%\n"
            )

        return wins, mean_scores

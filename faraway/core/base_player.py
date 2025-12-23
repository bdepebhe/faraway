from abc import ABC, abstractmethod

import numpy as np

from faraway.core.data_structures import MainCard


class BasePlayer(ABC):
    def __init__(
        self,
        n_rounds: int,
        n_cards_hand: int = 3,
        use_bonus_cards: bool = True,
    ):
        self.n_rounds = n_rounds
        self.use_bonus_cards = use_bonus_cards
        if self.use_bonus_cards:
            self.n_bonus_cards = n_rounds - 1
        else:
            self.n_bonus_cards = 0
        self.n_cards_hand = n_cards_hand
        self.cards_hand: np.ndarray = np.zeros((0, n_cards_hand, MainCard.length()))

    def reset_games_batch(self, batch_size: int) -> None:
        main_fields = np.zeros(
            (batch_size, self.n_rounds, MainCard.length())
        )  # (batch, n_main_cards, MainCard.length())
        bonus_fields = np.zeros(
            (batch_size, self.n_bonus_cards, MainCard.length())
        )  # (batch, n_bonus_cards, MainCard.length())
        self.fields = {
            "main": main_fields,
            "bonus": bonus_fields,
        }

    def get_current_batch_size(self) -> int:
        return int(self.fields["main"].shape[0])

    @abstractmethod
    def evaluate_cards(
        self,
        possible_cards: np.ndarray,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate possible cards and select one.

        Args:
            possible_cards: np.ndarray of shape (batch, n_cards, card_length)
            round_index: Current round index
            mode: "play", "draft", or "bonus"
            games_indices: Optional slice or indices to select specific batch elements from fields.
                           If None, uses all batch elements. Use slice(i, i+1) for single element.
        """
        pass

    def play_main_card(self, selected_cards: np.ndarray, round_index: int) -> None:
        self.fields["main"][:, round_index, :] = selected_cards

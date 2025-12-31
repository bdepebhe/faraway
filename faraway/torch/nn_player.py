from abc import abstractmethod
from typing import Any

import torch
import torch.nn as nn

from faraway.core.base_player import BasePlayer


class BaseNNPlayer(BasePlayer):
    def __init__(
        self,
        n_rounds: int,
        device: torch.device | None = None,
        model: nn.Module | None = None,
        model_params: dict[str, Any] | None = None,
        n_cards_hand: int = 3,
        use_bonus_cards: bool = True,
        use_cards_hand_in_state: bool = False,
    ):
        super().__init__(n_rounds, n_cards_hand, use_bonus_cards)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_cards_hand_in_state = use_cards_hand_in_state

        self.model_params = model_params or {}

    def set_model(self, model: nn.Module | None) -> None:
        if model is None:
            self.reset_model()
        else:
            self.model = model

    @abstractmethod
    def reset_model(self) -> None:
        pass

    def reset_games_batch(self, batch_size: int) -> None:
        super().reset_games_batch(batch_size)
        self.fields = {
            "main": torch.Tensor(self.fields["main"], device=self.device),
            "bonus": torch.Tensor(self.fields["bonus"], device=self.device),
        }

    @abstractmethod
    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def play_main_card(self, selected_cards: torch.Tensor, round_index: int) -> None:
        self.fields["main"][:, round_index, :] = selected_cards

    @abstractmethod
    def dump(self, path: str) -> None:
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseNNPlayer":
        pass

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
    ):
        super().__init__(n_rounds, n_cards_hand, use_bonus_cards)
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model_params = model_params or {}

        self.n_training_games_played = 0

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
            "main": torch.tensor(self.fields["main"], dtype=torch.float32, device=self.device),
            "bonus": torch.tensor(self.fields["bonus"], dtype=torch.float32, device=self.device),
        }

    @abstractmethod
    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
        return_logits: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate candidate cards and select one.

        Args:
            possible_cards_tensor: (batch, n_cards, card_length) candidate cards
            round_index: Current round index
            mode: "play", "draft", or "bonus"
            games_indices: Optional slice to select specific batch elements
            return_logits: If True, return logits instead of probabilities as first element
            temperature: Softmax temperature for exploration (>1 = more exploration)

        Returns:
            (probabilities_or_logits, selected_index, selected_cards)
        """
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

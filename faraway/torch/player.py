import torch
import torch.nn as nn

from faraway.core.data_structures import MainCard


class NNPlayer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        batch_size: int,
        n_main_cards: int,
        n_bonus_cards: int,
    ):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.n_main_cards = n_main_cards
        self.n_bonus_cards = n_bonus_cards
        self.reset_games_batch()

    def reset_games_batch(self) -> None:
        main_fields = torch.zeros(
            self.batch_size, self.n_main_cards, MainCard.length(), device=self.device
        )  # (batch, n_main_cards, MainCard.length())
        bonus_fields = torch.zeros(
            self.batch_size, self.n_bonus_cards, MainCard.length(), device=self.device
        )  # (batch, n_bonus_cards, MainCard.length())
        self.fields = {
            "main": main_fields,
            "bonus": bonus_fields,
        }

from typing import Any

import torch
import torch.nn as nn

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import MainCard
from faraway.torch.models import create_mlp_model


class NNPlayer(BasePlayer):
    def __init__(
        self,
        n_rounds: int,
        device: torch.device | None = None,
        model: nn.Module | None = None,
        model_params: dict[str, Any] | None = None,
        n_cards_hand: int = 3,
        use_bonus_cards: bool = True,
        use_cards_hand_in_state: bool = False,
        use_draft_indicator_in_model_input: bool = False,
    ):
        super().__init__(n_rounds, model_params, n_cards_hand, use_bonus_cards)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_length = (
            MainCard.length() * (self.n_rounds + self.n_bonus_cards)  # previous cards
            + 1  # round index
        )
        self.use_cards_hand_in_state = use_cards_hand_in_state
        if use_cards_hand_in_state:
            self.state_length += MainCard.length() * (n_cards_hand - 1)
        self.nn_input_size = self.state_length + MainCard.length()  # current card to choose
        self.use_draft_indicator_in_model_input = use_draft_indicator_in_model_input
        if use_draft_indicator_in_model_input:
            self.nn_input_size += 1  # draft indicator

        self.model_params = model_params or {}
        if model is None:
            self.reset_model()
        else:
            self.model = model
        # check that the model input size matches the nn_input_size
        model_input_size = self.model[0].in_features  # first layer is Linear
        if model_input_size != self.nn_input_size:
            raise ValueError(
                f"Model input size {model_input_size} does not match expected "
                f"input size {self.nn_input_size}"
            )

    def reset_model(self) -> None:
        self.model = create_mlp_model(self.nn_input_size, **self.model_params)

    def reset_games_batch(self, batch_size: int) -> None:
        super().reset_games_batch(batch_size)
        self.fields = {
            "main": torch.Tensor(self.fields["main"], device=self.device),
            "bonus": torch.Tensor(self.fields["bonus"], device=self.device),
        }

    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate possible cards and select one.

        Args:
            possible_cards_tensor: Tensor of shape (batch, n_cards, card_length)
            round_index: Current round index
            mode: "play", "draft", or "bonus"
            games_indices: Optional slice or indices to select specific batch elements from fields.
                           If None, uses all batch elements. Use slice(i, i+1) for single element.
        """
        if games_indices is None:
            games_indices = slice(None)
        batch_size = possible_cards_tensor.shape[0]
        fields = torch.concat(
            [self.fields["main"][games_indices], self.fields["bonus"][games_indices]], dim=1
        )
        flattened_state = torch.flatten(fields, start_dim=1)  # (batch, (8+6)*24)
        # append the round index
        round_index_tensor = torch.tensor([[round_index] * batch_size], device=self.device).T
        if self.use_draft_indicator_in_model_input:
            indicator = 1 if mode == "draft" else 0
            draft_indicator_tensor = torch.tensor([[indicator] * batch_size], device=self.device).T
            round_index_tensor = torch.concat([round_index_tensor, draft_indicator_tensor], dim=1)
        state_tensor = torch.concat(
            [flattened_state, round_index_tensor], dim=1
        )  # (batch, (8+6)*24 + 1)
        # expand the state tensor for each possible card
        expanded_state_tensor = state_tensor.unsqueeze(1).expand(
            -1, possible_cards_tensor.shape[1], -1
        )  # (batch, draft_size, (8+6)*24 + 1)
        # concatenate the state tensor with the possible cards tensor
        input_tensor = torch.concat(
            [expanded_state_tensor, possible_cards_tensor], dim=2
        )  # (batch, draft_size, (8+6)*24 + 1 + MainCard.length())
        # pass the input tensor through the model
        logits = self.model(input_tensor).squeeze(dim=2)  # (batch, draft_size)
        # take softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)  # (batch, draft_size)
        # sample from probabilities
        index = torch.multinomial(probabilities, 1)  # (batch,)
        # add the played card to the main field
        index_expanded = index.unsqueeze(2).expand(
            -1, -1, possible_cards_tensor.shape[2]
        )  # (batch, 1, card_length)
        selected_cards = torch.gather(possible_cards_tensor, 1, index_expanded).squeeze(
            1
        )  # (batch, card_length)
        self.cards_hand_index_to_replace = index
        return probabilities, index, selected_cards

    def play_main_card(self, selected_cards: torch.Tensor, round_index: int) -> None:
        self.fields["main"][:, round_index, :] = selected_cards

    def dump(self, path: str) -> None:
        """Save the player (model + config) to a single file."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_params": self.model_params,
            "config": {
                "n_rounds": self.n_rounds,
                "use_bonus_cards": self.use_bonus_cards,
                "use_cards_hand_in_state": self.use_cards_hand_in_state,
                "n_cards_hand": self.n_cards_hand,
                "use_draft_indicator_in_model_input": self.use_draft_indicator_in_model_input,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
    ) -> "NNPlayer":
        """Load a player from a saved checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Create the player with saved config
        player = cls(
            device=device,
            model_params=checkpoint["model_params"],
            **checkpoint["config"],
        )

        # Load the model weights
        player.model.load_state_dict(checkpoint["model_state_dict"])

        return player

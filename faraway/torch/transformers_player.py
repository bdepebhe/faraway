from typing import Any

import torch
import torch.nn as nn

from faraway.core.data_structures import MainCard
from faraway.torch.models import TransformerCardModel
from faraway.torch.nn_player import BaseNNPlayer


class TransformersPlayer(BaseNNPlayer):
    """
    Transformer-based player for the Faraway card game.

    Uses self-attention to model card interactions on the board and scores
    candidate cards via dot product with the global board representation.

    Architecture:
    1. Embed all cards (24 -> embed_dim) using a shared linear layer
    2. Add positional encodings for each board position (turns 1-8, bonus slots)
    3. Self-attention between all cards on the board
    4. Average pool to get global board representation
    5. Score candidates via dot product with board representation
    6. Softmax + sampling for REINFORCE policy
    """

    def __init__(
        self,
        n_rounds: int,
        device: torch.device | None = None,
        model: nn.Module | None = None,
        model_params: dict[str, Any] | None = None,
        n_cards_hand: int = 3,
        use_bonus_cards: bool = True,
        use_cards_hand_in_state: bool = False,
        use_mode_embedding: bool = False,
    ):
        super().__init__(
            n_rounds,
            device,
            model,
            model_params,
            n_cards_hand,
            use_bonus_cards,
            use_cards_hand_in_state,
        )

        self.use_mode_embedding = use_mode_embedding

        # Default model parameters
        self.model_params = model_params or {
            "embed_dim": 64,
            "n_attention_heads": 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }

        # Set model (will call reset_model if model is None)
        self.set_model(model)

    def reset_model(self) -> None:
        """Initialize a fresh TransformerCardModel."""
        self.model = TransformerCardModel(
            card_dim=MainCard.length(),
            n_main_positions=self.n_rounds,
            n_bonus_positions=self.n_bonus_cards,
            use_bonus_cards=self.use_bonus_cards,
            use_mode_embedding=self.use_mode_embedding,
            **self.model_params,
        ).to(self.device)

    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate possible cards and select one using the Transformer model.

        Args:
            possible_cards_tensor: Tensor of shape (batch, n_cards, card_length)
            round_index: Current round index (0-7)
            mode: "play", "draft", or "bonus" - affects model behavior via mode embedding
            games_indices: Optional slice or indices to select specific batch elements from fields.
                           If None, uses all batch elements.

        Returns:
            probabilities: (batch, n_cards) - probability distribution over candidates
            index: (batch, 1) - index of selected card
            selected_cards: (batch, card_length) - the selected card tensors
        """
        if games_indices is None:
            games_indices = slice(None)

        # Clone and detach all inputs to create completely independent tensors.
        # This avoids in-place modification issues during backward when fields
        # are updated between rounds. Gradients still flow through model weights.
        main_field = self.fields["main"][games_indices].clone().detach()  # (batch, 8, 24)
        bonus_field = (
            self.fields["bonus"][games_indices].clone().detach() if self.use_bonus_cards else None
        )  # (batch, 6, 24) or None
        candidate_cards = possible_cards_tensor.clone().detach()  # (batch, n_candidates, 24)

        # Forward pass through transformer (mode embedding distinguishes play/draft/bonus)
        logits = self.model(
            main_field=main_field,
            bonus_field=bonus_field,
            candidate_cards=candidate_cards,
            round_index=round_index,
            mode=mode,
        )  # (batch, n_candidates)

        # Softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)  # (batch, n_candidates)

        # Sample from the distribution (for REINFORCE)
        index = torch.multinomial(probabilities, 1)  # (batch, 1)

        # Gather the selected cards (already detached via candidate_cards)
        index_expanded = index.unsqueeze(2).expand(
            -1, -1, candidate_cards.shape[2]
        )  # (batch, 1, card_length)
        selected_cards = torch.gather(candidate_cards, 1, index_expanded).squeeze(
            1
        )  # (batch, card_length)

        # Store for potential hand management
        self.cards_hand_index_to_replace = index

        return probabilities, index, selected_cards

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
                "use_mode_embedding": self.use_mode_embedding,
            },
            "n_training_games_played": self.n_training_games_played,
            "player_type": "transformer",
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str,
        device: torch.device | None = None,
    ) -> "TransformersPlayer":
        """Load a player from a saved checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Extract config with backward compatibility for use_mode_embedding
        config = checkpoint["config"].copy()
        config.setdefault("use_mode_embedding", False)

        # Create the player with saved config
        player = cls(
            device=device,
            model_params=checkpoint["model_params"],
            **config,
        )

        # Load the model weights
        player.model.load_state_dict(checkpoint["model_state_dict"])
        player.n_training_games_played = checkpoint.get("n_training_games_played", 0)
        return player

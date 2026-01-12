"""NN models for the Faraway game."""

import torch
import torch.nn as nn


def create_mlp_model(
    input_size: int,
    hidden_layers_sizes: list[int] = [512, 512],  # noqa: B006
    dropout_rate: float = 0.1,
) -> nn.Module:
    layers_sizes = [input_size] + hidden_layers_sizes
    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(layers_sizes[-1], 1))
    return nn.Sequential(*layers)


class TransformerCardModel(nn.Module):  # type: ignore[misc]
    """
    Transformer-based model for card game decision making.

    Architecture:
    1. Card Embedding: Linear(card_dim -> embed_dim) for each card position
    2. Positional Encoding: Learnable embeddings for each position (turn 1-8)
    3. Mode Embedding: Learnable embeddings for play/draft/bonus modes
    4. Self-Attention: MultiheadAttention for card interactions
    5. Global Pooling: Average the card embeddings to get board summary
    6. Candidate Scoring: Dot product between board summary and embedded candidates

    The model processes the current board state (played cards) and scores
    candidate cards to determine which one to play.
    """

    # Mode string to index mapping
    MODE_TO_IDX = {"play": 0, "draft": 1, "bonus": 2}

    def __init__(
        self,
        card_dim: int = 24,
        embed_dim: int = 64,
        n_main_positions: int = 8,
        n_bonus_positions: int = 6,
        n_attention_heads: int = 4,
        n_transformer_layers: int = 2,
        dropout_rate: float = 0.1,
        use_bonus_cards: bool = True,
        use_mode_embedding: bool = False,
    ):
        super().__init__()
        self.card_dim = card_dim
        self.embed_dim = embed_dim
        self.n_main_positions = n_main_positions
        self.n_bonus_positions = n_bonus_positions if use_bonus_cards else 0
        self.n_total_positions = n_main_positions + self.n_bonus_positions
        self.use_bonus_cards = use_bonus_cards
        self.use_mode_embedding = use_mode_embedding

        # Card embedding layer (shared for all cards: main, bonus, and candidates)
        self.card_embedding = nn.Linear(card_dim, embed_dim)

        # Learnable positional encodings for board positions
        # Position 0-7: main cards (turns 1-8)
        # Position 8-13: bonus cards (if used)
        self.position_embedding = nn.Embedding(self.n_total_positions, embed_dim)

        # Round embedding (tells the model which round we're in)
        self.round_embedding = nn.Embedding(n_main_positions, embed_dim)

        # Mode embedding (play=0, draft=1, bonus=2)
        # Helps model distinguish the decision context
        if use_mode_embedding:
            self.mode_embedding = nn.Embedding(3, embed_dim)
        else:
            self.mode_embedding = None

        # Transformer encoder for self-attention between cards
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_attention_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # Layer norm for the global board representation
        self.board_layer_norm = nn.LayerNorm(embed_dim)

        # Projection for scoring (transforms board representation for dot product)
        self.score_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def embed_cards(self, cards: torch.Tensor) -> torch.Tensor:
        """Embed cards using the shared card embedding layer.

        Args:
            cards: (batch, n_cards, card_dim)

        Returns:
            (batch, n_cards, embed_dim)
        """
        return self.card_embedding(cards)

    def get_board_representation(
        self,
        main_field: torch.Tensor,
        bonus_field: torch.Tensor | None,
        round_index: int,
        mode: str = "play",
    ) -> torch.Tensor:
        """Compute the global board representation using self-attention.

        Args:
            main_field: (batch, n_main_positions, card_dim) - played main cards
            bonus_field: (batch, n_bonus_positions, card_dim) - played bonus cards (or None)
            round_index: current round (0-7)
            mode: decision context - "play", "draft", or "bonus"

        Returns:
            (batch, embed_dim) - global board summary vector
        """
        device = main_field.device

        # Embed main cards
        main_embedded = self.embed_cards(main_field)  # (batch, 8, embed_dim)

        # Add positional embeddings for main cards
        main_positions = torch.arange(self.n_main_positions, device=device)
        main_pos_embed = self.position_embedding(main_positions)  # (8, embed_dim)
        main_embedded = main_embedded + main_pos_embed.unsqueeze(0)

        if self.use_bonus_cards and bonus_field is not None:
            # Embed bonus cards
            bonus_embedded = self.embed_cards(bonus_field)  # (batch, 6, embed_dim)

            # Add positional embeddings for bonus cards
            bonus_positions = torch.arange(
                self.n_main_positions,
                self.n_total_positions,
                device=device,
            )
            bonus_pos_embed = self.position_embedding(bonus_positions)  # (6, embed_dim)
            bonus_embedded = bonus_embedded + bonus_pos_embed.unsqueeze(0)

            # Concatenate main and bonus
            board_embedded = torch.cat(
                [main_embedded, bonus_embedded], dim=1
            )  # (batch, 14, embed_dim)
        else:
            board_embedded = main_embedded

        # Add round information to all positions
        round_tensor = torch.tensor([round_index], device=device)
        round_embed = self.round_embedding(round_tensor)  # (1, embed_dim)
        board_embedded = board_embedded + round_embed.unsqueeze(0)

        # Add mode information to all positions (if enabled)
        if self.use_mode_embedding and self.mode_embedding is not None:
            mode_idx = self.MODE_TO_IDX.get(mode, 0)
            mode_tensor = torch.tensor([mode_idx], device=device)
            mode_embed = self.mode_embedding(mode_tensor)  # (1, embed_dim)
            board_embedded = board_embedded + mode_embed.unsqueeze(0)

        # Create attention mask to ignore empty card slots
        # Cards with all zeros are considered empty
        if self.use_bonus_cards and bonus_field is not None:
            all_cards = torch.cat([main_field, bonus_field], dim=1)
        else:
            all_cards = main_field
        # A card is empty if all features are 0
        card_mask = all_cards.abs().sum(dim=-1) == 0  # (batch, n_positions)
        # For transformer: True means "ignore this position"

        # Self-attention between cards
        board_attended = self.transformer_encoder(
            board_embedded,
            src_key_padding_mask=card_mask,
        )  # (batch, n_positions, embed_dim)

        # Global pooling: average non-empty positions
        # Mask for averaging (invert the padding mask)
        valid_mask = ~card_mask  # (batch, n_positions)
        valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)

        # Zero out padded positions before averaging
        board_attended = board_attended * valid_mask.unsqueeze(-1)
        board_summary = board_attended.sum(dim=1) / valid_counts  # (batch, embed_dim)

        # Layer norm for stability
        board_summary = self.board_layer_norm(board_summary)

        return board_summary

    def score_candidates(
        self,
        board_summary: torch.Tensor,
        candidate_cards: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate cards against the board summary.

        Args:
            board_summary: (batch, embed_dim) - global board representation
            candidate_cards: (batch, n_candidates, card_dim) - cards to choose from

        Returns:
            (batch, n_candidates) - score for each candidate
        """
        # Embed candidates
        candidates_embedded = self.embed_cards(candidate_cards)  # (batch, n_candidates, embed_dim)

        # Project board summary for scoring
        board_projected = self.score_projection(board_summary)  # (batch, embed_dim)

        # Dot product scoring: (batch, n_candidates, embed_dim) @ (batch, embed_dim, 1)
        scores = torch.bmm(candidates_embedded, board_projected.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, n_candidates)

        return scores

    def forward(
        self,
        main_field: torch.Tensor,
        bonus_field: torch.Tensor | None,
        candidate_cards: torch.Tensor,
        round_index: int,
        mode: str = "play",
    ) -> torch.Tensor:
        """Full forward pass: embed board, attend, and score candidates.

        Args:
            main_field: (batch, 8, card_dim) - played main cards
            bonus_field: (batch, 6, card_dim) - played bonus cards (or None)
            candidate_cards: (batch, n_candidates, card_dim) - cards to choose from
            round_index: current round (0-7)
            mode: decision context - "play", "draft", or "bonus"

        Returns:
            (batch, n_candidates) - logits for each candidate card
        """
        board_summary = self.get_board_representation(main_field, bonus_field, round_index, mode)
        logits = self.score_candidates(board_summary, candidate_cards)
        return logits

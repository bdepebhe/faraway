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


# =============================================================================
# Tensor field indices (matching MainCard.flatten() layout)
# =============================================================================
# Position 0: id
# Positions 1-9: assets (rock, animal, vegetal, red, green, blue, yellow, night, map)
# Positions 10-20: rewards (rock, animal, vegetal, red, green, blue, yellow, night,
# map, all_4_colors, flat)
# Positions 21-23: prerequisites (rock, animal, vegetal)

IDX_ID = 0
IDX_ASSETS_START = 1
IDX_ASSETS_END = 10  # exclusive
IDX_REWARDS_START = 10
IDX_REWARDS_END = 21  # exclusive
IDX_PREREQ_START = 21
IDX_PREREQ_END = 24  # exclusive

# Asset indices (relative to IDX_ASSETS_START)
ASSET_ROCK = 0
ASSET_ANIMAL = 1
ASSET_VEGETAL = 2
ASSET_RED = 3
ASSET_GREEN = 4
ASSET_BLUE = 5
ASSET_YELLOW = 6
ASSET_NIGHT = 7
ASSET_MAP = 8
N_ASSETS = 9

# Reward indices (relative to IDX_REWARDS_START)
REWARD_ALL_4_COLORS = 9  # all_4_colors
REWARD_FLAT = 10  # flat
N_REWARDS = 11


def final_count_from_tensor_field_legacy(
    main_field_tensor: torch.Tensor, bonus_field_tensor: torch.Tensor
) -> torch.Tensor:
    """Legacy scoring using Python/Pydantic objects (for testing/comparison)."""
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


def final_count_tensor_batched(
    main_field: torch.Tensor,
    bonus_field: torch.Tensor,
) -> torch.Tensor:
    """
    Pure tensor implementation of final_count for batched games.

    Args:
        main_field: (batch, n_rounds, 24) - played main cards
        bonus_field: (batch, n_bonus, 24) - played bonus cards (may include empty slots)

    Returns:
        (batch,) - total score for each game

    The scoring algorithm:
    1. Main cards are evaluated from last to first
    2. When evaluating a card, the "visible" assets are:
       - All cards from that position to the last position (in original order)
       - Plus all bonus cards
    3. Prerequisites are checked: rock/animal/vegetal assets >= prerequisites
    4. If prerequisites met: reward = sum(visible_assets * reward_multipliers)
    5. Bonus cards are evaluated with all assets visible (no prerequisites)
    """
    batch_size = main_field.shape[0]
    n_rounds = main_field.shape[1]
    device = main_field.device

    # Extract components from main cards
    # main_assets: (batch, n_rounds, 9)
    main_assets = main_field[:, :, IDX_ASSETS_START:IDX_ASSETS_END]
    # main_rewards: (batch, n_rounds, 11)
    main_rewards = main_field[:, :, IDX_REWARDS_START:IDX_REWARDS_END]
    # main_prereqs: (batch, n_rounds, 3) - rock, animal, vegetal only
    main_prereqs = main_field[:, :, IDX_PREREQ_START:IDX_PREREQ_END]

    # Extract bonus card components (filter out empty slots where id == 0)
    # bonus_assets: (batch, n_bonus, 9)
    bonus_assets = bonus_field[:, :, IDX_ASSETS_START:IDX_ASSETS_END]
    # bonus_rewards: (batch, n_bonus, 11)
    bonus_rewards = bonus_field[:, :, IDX_REWARDS_START:IDX_REWARDS_END]
    # bonus_valid: (batch, n_bonus) - True if card is actually played (id > 0)
    bonus_valid = bonus_field[:, :, IDX_ID] > 0

    # Sum all bonus assets (they're always visible): (batch, 9)
    bonus_assets_sum = (bonus_assets * bonus_valid.unsqueeze(-1)).sum(dim=1)

    # Compute cumulative assets from each position to the end (reversed cumsum)
    # For position i, we want sum of assets from i to n_rounds-1
    # cumsum on reversed, then reverse back
    # main_assets_reversed: (batch, n_rounds, 9) with cards in reverse order
    main_assets_reversed = main_assets.flip(dims=[1])
    # cumsum_reversed: (batch, n_rounds, 9)
    cumsum_reversed = main_assets_reversed.cumsum(dim=1)
    # cumsum_from_end: (batch, n_rounds, 9) - at position i, sum of assets from i to end
    cumsum_from_end = cumsum_reversed.flip(dims=[1])

    # Total visible assets at each position: main cumsum + all bonus assets
    # visible_assets: (batch, n_rounds, 9)
    visible_assets = cumsum_from_end + bonus_assets_sum.unsqueeze(1)

    # Compute derived assets for reward calculation
    # all_4_colors = min(red, green, blue, yellow)
    # Indices: red=3, green=4, blue=5, yellow=6 (relative to assets)
    color_assets = visible_assets[:, :, ASSET_RED : ASSET_YELLOW + 1]  # (batch, n_rounds, 4)
    all_4_colors = color_assets.min(dim=2).values  # (batch, n_rounds)

    # flat = 1 (constant)
    flat_asset = torch.ones(batch_size, n_rounds, device=device)

    # Build extended assets for reward computation: (batch, n_rounds, 11)
    # Rewards expect: rock, animal, vegetal, red, green, blue, yellow, night, map,
    # all_4_colors, flat
    extended_assets = torch.cat(
        [
            visible_assets,  # (batch, n_rounds, 9) - rock through map
            all_4_colors.unsqueeze(-1),  # (batch, n_rounds, 1)
            flat_asset.unsqueeze(-1),  # (batch, n_rounds, 1)
        ],
        dim=2,
    )  # (batch, n_rounds, 11)

    # Check prerequisites: assets[rock,animal,vegetal] >= prerequisites[rock,animal,vegetal]
    # prereq_assets: (batch, n_rounds, 3)
    prereq_assets = visible_assets[:, :, ASSET_ROCK : ASSET_VEGETAL + 1]
    # prereqs_met: (batch, n_rounds) - all 3 conditions must be true
    prereqs_met = (prereq_assets >= main_prereqs).all(dim=2)

    # Compute main card rewards: dot product of extended_assets and main_rewards
    # (batch, n_rounds, 11) * (batch, n_rounds, 11) -> sum -> (batch, n_rounds)
    main_card_rewards = (extended_assets * main_rewards).sum(dim=2)

    # Mask by prerequisites: only count rewards where prerequisites are met
    main_card_rewards_masked = main_card_rewards * prereqs_met.float()

    # Sum main card rewards: (batch,)
    total_main_rewards = main_card_rewards_masked.sum(dim=1)

    # Compute bonus card rewards
    # Bonus cards use the final assets (all main cards + all bonus cards visible)
    # final_assets: (batch, 9) - sum of all main and bonus assets
    final_main_assets = main_assets.sum(dim=1)  # (batch, 9)
    final_assets = final_main_assets + bonus_assets_sum  # (batch, 9)

    # Compute derived assets for bonus rewards
    final_colors = final_assets[:, ASSET_RED : ASSET_YELLOW + 1]  # (batch, 4)
    final_all_4_colors = final_colors.min(dim=1).values  # (batch,)
    final_flat = torch.ones(batch_size, device=device)

    # Extended final assets: (batch, 11)
    final_extended_assets = torch.cat(
        [
            final_assets,  # (batch, 9)
            final_all_4_colors.unsqueeze(-1),  # (batch, 1)
            final_flat.unsqueeze(-1),  # (batch, 1)
        ],
        dim=1,
    )

    # Bonus rewards: (batch, n_bonus, 11) * (batch, 1, 11) -> sum -> (batch, n_bonus)
    bonus_card_rewards = (bonus_rewards * final_extended_assets.unsqueeze(1)).sum(dim=2)

    # Mask by valid bonus cards
    bonus_card_rewards_masked = bonus_card_rewards * bonus_valid.float()

    # Sum bonus card rewards: (batch,)
    total_bonus_rewards = bonus_card_rewards_masked.sum(dim=1)

    # Total score
    return total_main_rewards + total_bonus_rewards


def final_count_from_tensor_field(
    main_field_tensor: torch.Tensor, bonus_field_tensor: torch.Tensor
) -> torch.Tensor:
    """
    Compute final score from tensor fields (single game, for backward compatibility).

    Args:
        main_field_tensor: (n_rounds, 24) - played main cards for one game
        bonus_field_tensor: (n_bonus, 24) - played bonus cards for one game

    Returns:
        Scalar tensor with the score
    """
    # Add batch dimension and use batched implementation
    main_batched = main_field_tensor.unsqueeze(0)
    bonus_batched = bonus_field_tensor.unsqueeze(0)
    return final_count_tensor_batched(main_batched, bonus_batched)[0]


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
        # Track discarded bonus cards (drawn but not chosen) for reshuffling
        # Per official rules: when deck is empty, shuffle discarded cards to form new deck
        self.bonus_discard = torch.zeros(
            batch_size, self.decks["bonus"].shape[0], dtype=torch.bool, device=self.device
        )  # (batch, n_total_bonus_cards) - True means card is in discard pile
        for player in self.players:
            player.reset_games_batch(batch_size)
        self.round_index = 0

    def deal_initial_hands(
        self,
        players: Sequence[BasePlayer] | None = None,
        n_cards: int = 3,
        batch_size: int | None = None,
    ) -> None:
        """Deal initial hands to players from the main deck.

        Args:
            players: List of players to deal hands to. Defaults to self.players.
            n_cards: Number of cards per hand. Defaults to 3.
            batch_size: Batch size. If None, inferred from deck availability.
        """
        if players is None:
            players = self.players
        if batch_size is None:
            batch_size = self.deck_availability["main"].shape[0]

        card_length = self.decks["main"].shape[1]
        expanded_main_deck = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)

        for player in players:
            # Draw n_cards from the deck
            indices = torch.multinomial(
                self.deck_availability["main"].float(), n_cards, replacement=False
            )  # (batch, n_cards)

            # Expand indices for gather: (batch, n_cards) -> (batch, n_cards, card_length)
            indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)

            # Gather cards: (batch, deck_size, card_length) -> (batch, n_cards, card_length)
            player.cards_hand = torch.gather(expanded_main_deck, 1, indices_expanded)

            # Mark these cards as unavailable in the deck
            self.deck_availability["main"].scatter_(1, indices, False)

    def get_used_cards_ids(self, type: str, game_id: int) -> list[int]:
        used_cards_ids = [
            1 + card_id
            for card_id in torch.where(~self.deck_availability[type][game_id])[0].tolist()
        ]
        return used_cards_ids

    def reshuffle_bonus_discard_if_needed(
        self, game_ids: torch.Tensor | None = None, n_cards_needed: int = 1
    ) -> None:
        """Reshuffle discarded bonus cards back into the deck when needed.

        Per official rules: "If the Sanctuary deck is empty, shuffle the discarded
        Sanctuary cards (those not chosen by players) to form a new deck."

        Args:
            game_ids: Tensor of game indices to check/reshuffle (None = all games)
            n_cards_needed: Number of cards needed for the draw
        """
        if game_ids is None:
            game_ids = torch.arange(self.deck_availability["bonus"].shape[0], device=self.device)

        # For each game, check if we need to reshuffle
        for game_id in game_ids:
            gid = int(game_id.item()) if isinstance(game_id, torch.Tensor) else game_id
            n_available = self.deck_availability["bonus"][gid].sum().item()

            if n_available < n_cards_needed:
                # Reshuffle: move discarded cards back to available
                n_discarded = self.bonus_discard[gid].sum().item()
                if n_discarded > 0:
                    if self.verbose > 1:
                        logger.debug(
                            f"Game #{gid}: Reshuffling {int(n_discarded)} discarded bonus cards "
                            f"(had {int(n_available)} available, need {n_cards_needed})"
                        )
                    # Move discarded cards back to available
                    self.deck_availability["bonus"][gid] |= self.bonus_discard[gid]
                    # Clear the discard pile
                    self.bonus_discard[gid] = False

    def get_scores(self) -> torch.Tensor:
        """
        Compute final scores for all players in the batch.

        Uses the pure tensor implementation for efficiency.

        Returns:
            Tensor of shape (batch, n_players) with final scores.
        """
        return torch.stack(
            [
                final_count_tensor_batched(
                    player.fields["main"],
                    player.fields["bonus"],
                )
                for player in self.players
            ],
            dim=1,
        )  # (batch, players)

    def get_bonus_cards_played(self) -> torch.Tensor:
        """Count the number of bonus cards played by each player.

        A bonus card slot is considered "played" if its card ID (first element) is > 0.
        Maximum is n_rounds - 1 (7 for 8 rounds, since no bonus on first round).

        Returns:
            Tensor of shape (batch, n_players) with counts of bonus cards played.
        """
        return torch.stack(
            [
                # Card ID is in position 0; a card is played if ID > 0
                (player.fields["bonus"][:, :, 0] > 0).sum(dim=1).float()
                for player in self.players
            ],
            dim=1,
        )  # (batch, n_players)

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
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """Run a tournament and return wins, mean scores, bonus cards,
        and draft priority per player.

        Args:
            n_batches: Number of batches to play
            batch_size: Number of games per batch
            player_names: Optional names for TensorBoard logging

        Returns:
            Tuple of (wins, mean scores, mean bonus cards, mean draft priority rate) per player.
            Draft priority rate is 0.0 for base implementation (only tracked in RealNNGame).
        """
        # Collect scores and bonus cards from each batch
        all_scores: list[torch.Tensor] = []
        all_bonus_cards: list[torch.Tensor] = []

        for i in range(n_batches):
            self.play_games_batch(batch_size)
            if self.verbose > 1:
                logger.info(f"Batch {i + 1} completed")
            all_scores.append(self.get_scores())
            if self.use_bonus_cards:
                all_bonus_cards.append(self.get_bonus_cards_played())

        scores = torch.cat(all_scores, dim=0)  # (n_batches * batch_size, players)
        self.total_games_played += n_batches * batch_size

        winner = scores.argmax(dim=1)
        wins = []
        win_rate = []
        for player_id in range(len(self.players)):
            wins.append(torch.where(winner == player_id)[0].shape[0])
            win_rate.append(wins[-1] / (n_batches * batch_size) * 100)

        mean_scores = scores.mean(dim=0).tolist()

        # Compute mean bonus cards per player
        if self.use_bonus_cards and all_bonus_cards:
            bonus_cards = torch.cat(all_bonus_cards, dim=0)  # (total_games, players)
            mean_bonus_cards = bonus_cards.mean(dim=0).tolist()
        else:
            mean_bonus_cards = [0.0] * len(self.players)

        if self.verbose > 0:
            logger.info(
                f"Tournament completed.\n"
                f"Mean scores: {mean_scores}\n"
                f"Wins: {wins}\n"
                f"Win rate: {win_rate}%\n"
                f"Mean bonus cards: {mean_bonus_cards}\n"
            )

        # Base implementation doesn't track draft priority (only RealNNGame does)
        mean_draft_priority: list[float] = [0.0] * len(self.players)
        return wins, mean_scores, mean_bonus_cards, mean_draft_priority

"""
Pure tensor-based solo play for Faraway game.

No dependencies on legacy pydantic classes - everything is tensors.

Tensor representations:
- Main card: 24 features [id, assets(9), rewards(11), prerequisites(3)]
- Bonus card: 20 features [assets(9), rewards(11)]
- Player main field: (batch, 8, 24)
- Player bonus field: (batch, 7, 20)
- Deck availability tracked via index masks
"""

import sys
from typing import Annotated, Any

import torch
import typer
from loguru import logger

from faraway.torch.base_game import BaseNNGame
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.nn_player import BaseNNPlayer
from faraway.torch.play_vs_random import play_vs_random
from faraway.torch.transformers_player import TransformersPlayer


def sample_cards_from_availability_tensor(
    availability_tensor: torch.Tensor, draft_size: int
) -> torch.Tensor:
    # the availability tensor is a boolean tensor of shape (batch, n_cards)
    # we want to return a tensor of size (batch, draft_size) with the indices of the sampled cards
    # we can use torch.multinomial to sample the indices with replacement
    return torch.multinomial(availability_tensor, draft_size, replacement=False)


class SoloLearningGame(BaseNNGame):
    """
    Batched solo play game using tensors for REINFORCE-based training.

    Supports two advantage computation modes:
        - Baseline-based (default): advantage = reward - EMA_baseline
        - Peer-relative: advantage = (reward - batch_mean) / batch_std (z-score)
          or advantage = reward - batch_mean (center)

    Peer-relative reward mode (rl_params["peer_relative_reward"] = True):
        Games in a sub-batch see the exact same cards in the same order (fixed seed).
        This enables fair comparison: score differences are purely due to model decisions.
        Advantage is computed within each sub-batch (z-score or centering).
        Requires appropriate deck/draft sizes to avoid deck exhaustion (e.g., hand=3, draft=3).

        Sub-batches (rl_params["peer_sub_batch_size"]):
            The training batch can be divided into sub-batches. Each sub-batch gets a
            different deck shuffle, but games within the same sub-batch see identical cards.
            Example: batch_size=128, peer_sub_batch_size=32 → 4 sub-batches with different
            deck shuffles, advantage normalized within each 32-game sub-batch.
            If peer_sub_batch_size is None, the entire batch is one sub-batch.

        In this mode, prior_baseline_score and update_baseline_rate are NOT used for
        training (advantage comes from sub-batch mean), but the baseline EMA is still
        tracked for TensorBoard monitoring.

    Attributes:
        n_rounds: Number of rounds in the game
        draft_size: Number of cards shown in each draft
        peer_relative_reward: If True, use fixed seed and within-sub-batch advantage
        peer_sub_batch_size: Size of sub-batches for peer comparison (None = full batch)
        advantage_peer_normalization: "zscore" or "center" (only used if peer_relative_reward)
        device: Torch device for tensors

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
        draft_size: int = 10,
        replace_remaining_cards: bool = True,
        use_bonus_cards: bool = True,
        use_hand: bool = False,
        n_cards_hand: int = 3,
        model_path: str | None = None,
        verbose: int = 1,
        device: torch.device | None = None,
        model_params: dict[str, Any] | None = None,
        player_type: str = "mlp",
        player_params: dict[str, Any] | None = None,
        optimizer_params: dict[str, Any] | None = None,
        rl_params: dict[str, Any] | None = None,
        experiment_name: str | None = None,
        log_dir: str = "runs",
        eval_vs_random_config: dict[str, Any] | None = None,
        eval_solo_config: dict[str, Any] | None = None,
    ):
        super().__init__(
            n_rounds,
            use_bonus_cards,
            device,
            verbose=verbose,
            experiment_name=experiment_name,
            log_dir=log_dir,
        )
        self.draft_size = draft_size
        self.replace_remaining_cards = replace_remaining_cards
        self.use_hand = use_hand
        self.n_cards_hand = n_cards_hand
        self.model_params = model_params or {
            "hidden_layers_sizes": [512, 512],
            "dropout_rate": 0.1,
        }
        self.player_type = player_type
        self.player_params = player_params or {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
        self.optimizer_params = optimizer_params or {
            "lr": 0.001,
        }
        self.rl_params = rl_params or {
            "prior_baseline_score": 29,
            "train_batch_size": 32,
            "update_baseline_rate": 0.05,
        }
        # Peer-relative reward settings
        self.peer_relative_reward = self.rl_params.get("peer_relative_reward", False)
        self.advantage_peer_normalization = self.rl_params.get(
            "advantage_peer_normalization", "zscore"
        )  # "zscore" or "center"
        # Sub-batch size for peer comparison (games within a sub-batch share the same deck shuffle)
        # If None or 0, the entire batch is one sub-batch (all games see same cards)
        self.peer_sub_batch_size = self.rl_params.get("peer_sub_batch_size", None)
        self.player_params["use_bonus_cards"] = self.use_bonus_cards

        # Evaluation config
        self.eval_vs_random_config = eval_vs_random_config or {}
        self.eval_solo_config = eval_solo_config or {}

        # Initialize TensorBoard (uses base class method)
        self.init_tensorboard()
        self.reset_learning(model_path=model_path)
        self.players: list[BaseNNPlayer]

    def reset_learning(self, model_path: str | None = None) -> None:
        if model_path is not None:
            model = torch.load(model_path)
        else:
            model = None
        self.baseline = self.rl_params["prior_baseline_score"]
        self.step_id = 0  # step counter for evaluation frequency (session-local, not saved)

        if self.player_type == "mlp":
            self.players = [
                MLPPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]  # only one player for solo play
        elif self.player_type == "transformer":
            self.players = [
                TransformersPlayer(
                    model=model,
                    model_params=self.model_params,
                    device=self.device,
                    n_rounds=self.n_rounds,
                    **self.player_params,
                )
            ]  # only one player for solo play
        else:
            raise ValueError(f"Unknown player type: {self.player_type}")
        self.optimizer = torch.optim.Adam(
            self.players[0].model.parameters(), **self.optimizer_params
        )

    def reset_games_batch(self, batch_size: int) -> None:
        """Reset games and initialize player hands if using draft mode."""
        super().reset_games_batch(batch_size)

        if self.peer_relative_reward:
            # Pre-shuffle decks for peer comparison: games in same sub-batch see same cards
            self._setup_fixed_seed_decks(batch_size)

            if self.use_hand:
                # Deal same initial hand to games within each sub-batch
                self._deal_initial_hands_fixed_seed(
                    n_cards=self.n_cards_hand, batch_size=batch_size
                )
        elif self.use_hand:
            # Deal random initial hands (legacy behavior)
            self.deal_initial_hands(n_cards=self.n_cards_hand, batch_size=batch_size)

    def _setup_fixed_seed_decks(self, batch_size: int) -> None:
        """Pre-shuffle decks for peer-relative reward (fixed seed within sub-batches).

        Games within the same sub-batch see the same cards in the same order.
        Different sub-batches get different deck shuffles for diversity.
        """
        # Determine number of sub-batches
        sub_batch_size = self.peer_sub_batch_size or batch_size
        self.n_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size  # ceil division
        self.current_sub_batch_size = sub_batch_size

        # Create master deck orderings for each sub-batch
        self.master_deck_orders = {
            "main": [
                torch.randperm(self.decks["main"].shape[0], device=self.device)
                for _ in range(self.n_sub_batches)
            ],
            "bonus": [
                torch.randperm(self.decks["bonus"].shape[0], device=self.device)
                for _ in range(self.n_sub_batches)
            ],
        }
        # Track position in the master order for sequential dealing (per sub-batch)
        self.deck_cursors = {
            "main": [0] * self.n_sub_batches,
            "bonus": [0] * self.n_sub_batches,
        }

    def _compute_peer_relative_advantage(self, shaped_reward: torch.Tensor) -> torch.Tensor:
        """Compute advantage by normalizing within each sub-batch.

        Games in the same sub-batch faced identical cards, so their scores can be
        fairly compared. Normalization (z-score or centering) is done per sub-batch.

        Args:
            shaped_reward: Tensor of shape (batch,) with (shaped) rewards

        Returns:
            Tensor of shape (batch,) with peer-relative advantages
        """
        batch_size = shaped_reward.shape[0]

        # Reshape to (n_sub_batches, sub_batch_size) for vectorized normalization
        # Pad if batch_size is not evenly divisible
        padded_size = self.n_sub_batches * self.current_sub_batch_size
        if batch_size < padded_size:
            # Pad with zeros (won't affect mean/std of real data)
            padded_rewards = torch.zeros(padded_size, device=shaped_reward.device)
            padded_rewards[:batch_size] = shaped_reward
            reshaped = padded_rewards.view(self.n_sub_batches, self.current_sub_batch_size)
            # Create mask for valid entries
            valid_mask = torch.zeros(padded_size, dtype=torch.bool, device=shaped_reward.device)
            valid_mask[:batch_size] = True
            valid_mask = valid_mask.view(self.n_sub_batches, self.current_sub_batch_size)
        else:
            reshaped = shaped_reward.view(self.n_sub_batches, self.current_sub_batch_size)
            valid_mask = None

        # Compute mean per sub-batch: (n_sub_batches, 1)
        if valid_mask is not None:
            # Handle partial last sub-batch
            sub_means = (reshaped * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
        else:
            sub_means = reshaped.mean(dim=1, keepdim=True)

        if self.advantage_peer_normalization == "zscore":
            # Compute std per sub-batch
            centered = reshaped - sub_means
            if valid_mask is not None:
                sub_stds = (
                    (centered**2 * valid_mask).sum(dim=1, keepdim=True)
                    / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
                ).sqrt()
            else:
                sub_stds = reshaped.std(dim=1, keepdim=True)
            normalized = centered / (sub_stds + 1e-8)
        else:  # "center"
            normalized = reshaped - sub_means

        # Flatten and trim to original batch_size
        advantage = normalized.view(-1)[:batch_size]

        return advantage

    def _deal_initial_hands_fixed_seed(self, n_cards: int, batch_size: int) -> None:
        """Deal initial hands with fixed seed: games in same sub-batch get the same hand."""
        player = self.players[0]
        card_length = self.decks["main"].shape[1]
        expanded_main_deck = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)

        # Vectorized: all games in same sub-batch get same cards
        # Stack all deck orders: (n_sub_batches, deck_size)
        stacked_orders = torch.stack(self.master_deck_orders["main"])

        # Get cursor (same for all sub-batches)
        cursor = self.deck_cursors["main"][0]

        # Get indices for each sub-batch: (n_sub_batches, n_cards)
        sub_batch_indices = stacked_orders[:, cursor : cursor + n_cards]

        # Compute sub-batch ID for each game: (batch_size,)
        sub_batch_ids = torch.arange(batch_size, device=self.device) // self.current_sub_batch_size

        # Index to get indices for each game: (batch_size, n_cards)
        indices = sub_batch_indices[sub_batch_ids]

        # Update all cursors
        for i in range(self.n_sub_batches):
            self.deck_cursors["main"][i] += n_cards

        # Expand indices for gather: (batch, n_cards) -> (batch, n_cards, card_length)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)

        # Gather cards: (batch, deck_size, card_length) -> (batch, n_cards, card_length)
        player.cards_hand = torch.gather(expanded_main_deck, 1, indices_expanded)

        # Mark these cards as unavailable in the deck
        self.deck_availability["main"].scatter_(1, indices, False)

    def dump_model(self, model_path: str) -> None:
        torch.save(self.players[0].model, model_path)

    def dump_player(self, player_path: str) -> None:
        """Save the player (model + config) to a file."""
        self.players[0].dump(player_path)

    def dump_training_state(self, path: str) -> None:
        """Save training state (player + baseline) to a file."""
        checkpoint = {
            "player_state": self.players[0].model.state_dict(),
            "player_params": getattr(self.players[0], "model_params", {}),
            "player_config": {
                "n_rounds": self.players[0].n_rounds,
                "use_bonus_cards": self.players[0].use_bonus_cards,
                "n_cards_hand": self.players[0].n_cards_hand,
            },
            "n_training_games_played": self.players[0].n_training_games_played,
            "baseline": self.baseline,
            "player_type": self.player_type,
        }
        # Add transformer-specific config
        if hasattr(self.players[0], "use_mode_embedding"):
            checkpoint["player_config"]["use_mode_embedding"] = self.players[0].use_mode_embedding
        torch.save(checkpoint, path)

    def load_training_state(self, path: str) -> None:
        """Load training state (model weights + baseline) from a file."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.players[0].model.load_state_dict(checkpoint["player_state"])
        self.players[0].n_training_games_played = checkpoint.get("n_training_games_played", 0)
        if "baseline" in checkpoint:
            self.baseline = checkpoint["baseline"]
            logger.info(f"Restored baseline: {self.baseline:.2f}")
        logger.info(f"Restored n_training_games_played: {self.players[0].n_training_games_played}")

    def run_eval_vs_random(self) -> tuple[float, float]:
        """Run evaluation against random players using the shared TensorBoard writer."""
        n_random_players = self.eval_vs_random_config.get("n_players", 1)

        if self.verbose > 0:
            logger.info(f"Running eval vs {n_random_players} random player(s)...")
        win_rate, mean_score = play_vs_random(
            player=self.players[0],
            n_random_players=n_random_players,
            n_eval_batches=self.eval_vs_random_config.get("n_batches", 100),
            batch_size=self.eval_vs_random_config.get("batch_size", 32),
            writer=self.writer,  # Share the TensorBoard writer
            verbose=0,  # Quiet mode for intermediate evals
        )
        if self.verbose > 0:
            logger.info(f"Eval vs random: win_rate={win_rate:.2%}, mean_score={mean_score:.2f}")
        return win_rate, mean_score

    def run_eval_solo(self) -> float:
        """Run solo evaluation and log to TensorBoard."""
        n_batches = self.eval_solo_config.get("n_batches", 100)
        batch_size = self.eval_solo_config.get("batch_size", 32)

        if self.verbose > 0:
            logger.info(f"Running solo eval ({n_batches} batches x {batch_size})...")

        scores = self.play_games_batches(n_batches=n_batches, batch_size=batch_size)
        mean_score = scores.mean().item()

        if self.writer is not None:
            step = self.players[0].n_training_games_played
            self.writer.add_scalar("eval/solo/mean_score", mean_score, step)
            self.writer.add_scalar("eval/solo/max_score", scores.max().item(), step)
            self.writer.add_scalar("eval/solo/min_score", scores.min().item(), step)
            self.writer.add_scalar("eval/solo/std_score", scores.std().item(), step)
            self.writer.flush()

        if self.verbose > 0:
            logger.info(f"Eval solo: mean_score={mean_score:.2f}")

        return float(mean_score)

    def initialize_baseline_from_eval(
        self,
        n_batches: int | None = None,
        batch_size: int | None = None,
    ) -> float:
        """Initialize baseline from solo evaluation score.

        This runs a solo evaluation and sets the baseline to the mean score,
        providing a more accurate initial baseline than a hardcoded value.

        Args:
            n_batches: Number of evaluation batches (default: from eval_solo_config or 50)
            batch_size: Games per batch (default: from eval_solo_config or 64)

        Returns:
            The mean score used as the new baseline.
        """
        # Use provided values or fall back to eval_solo_config or defaults
        n_batches = n_batches or self.eval_solo_config.get("n_batches", 50)
        batch_size = batch_size or self.eval_solo_config.get("batch_size", 64)

        if self.verbose > 0:
            logger.info(
                f"Initializing baseline from solo eval ({n_batches} batches x {batch_size})..."
            )

        # Run evaluation without TensorBoard logging (just for baseline init)
        scores = self.play_games_batches(n_batches=n_batches, batch_size=batch_size)
        mean_score = scores.mean().item()

        # Set baseline
        old_baseline = self.baseline
        self.baseline = mean_score

        if self.verbose > 0:
            logger.info(
                f"Baseline initialized: {old_baseline:.2f} -> {self.baseline:.2f} "
                f"(from {n_batches * batch_size} games)"
            )

        return float(mean_score)

    def log_hparams(self, extra_hparams: dict[str, Any] | None = None) -> None:
        """Log hyperparameters to TensorBoard for experiment comparison."""
        # Count trainable parameters
        n_trainable_params = sum(
            p.numel() for p in self.players[0].model.parameters() if p.requires_grad
        )

        hparams = {
            "player_type": self.player_type,
            "n_rounds": self.n_rounds,
            "draft_size": self.draft_size,
            "use_hand": self.use_hand,
            "n_cards_hand": self.n_cards_hand,
            "n_trainable_params": n_trainable_params,
            "rl_params": self.rl_params,
            "model_params": self.model_params,
            "player_params": self.player_params,
            "optimizer_params": self.optimizer_params,
            "use_bonus_cards": self.use_bonus_cards,
            "replace_remaining_cards": self.replace_remaining_cards,
        }
        if extra_hparams:
            hparams.update(extra_hparams)
        if self.writer is not None:
            # Log trainable params as a scalar for easy comparison across runs
            self.writer.add_scalar("model/n_trainable_params", n_trainable_params, 0)
            # Use add_text instead of add_hparams to avoid creating timestamp subdirectories
            hparams_text = "\n".join(f"**{k}**: {v}" for k, v in hparams.items())
            self.writer.add_text("hparams", hparams_text, 0)

    def _play_from_hand(self) -> torch.Tensor:
        """Choose and play a card from hand. Returns probability of the selection.

        Used when use_hand=True. The model selects one card from the hand to play.
        """
        player = self.players[0]

        # Model chooses which card from hand to play (mode="play")
        probabilities, index, selected_cards = player.evaluate_cards(
            player.cards_hand, self.round_index, mode="play"
        )

        # Play the selected card
        player.play_main_card(selected_cards, self.round_index)

        # Store which hand slot was used (for replacement in _draft_to_hand)
        self.hand_slot_to_replace = index.squeeze(1)  # (batch,)

        # Get the probability of the selected card
        probability = torch.gather(probabilities, 1, index).squeeze(1)  # (batch,)
        return probability

    def _draft_to_hand(self) -> torch.Tensor:
        """Draft a card from the river to replace the played card in hand.

        Used when use_hand=True. The model selects one card from the river to add to hand.
        """
        player = self.players[0]
        batch_size = player.get_current_batch_size()

        # With peer_relative_reward and appropriate deck/draft sizes, deck won't be exhausted
        if not self.peer_relative_reward:
            # Check if there are enough cards in the deck (only needed for non-peer mode)
            available_counts = self.deck_availability["main"].sum(dim=1)  # (batch,)
            can_draft = available_counts >= self.draft_size

            if not can_draft.any():
                # No drafting possible (late game), return prob=1 (no decision)
                return torch.ones(batch_size, device=self.device)

        # Sample cards from the deck for the river
        indices = self._sample_draft_indices("main", batch_size)  # (batch, draft_size)

        # Get the actual card tensors
        river_cards = self.decks["main"][indices]  # (batch, draft_size, card_dim)

        # Model chooses which card to draft (mode="draft")
        probabilities, index, selected_cards = player.evaluate_cards(
            river_cards, self.round_index, mode="draft"
        )

        # Replace the used hand slot with the drafted card
        batch_indices = torch.arange(batch_size, device=self.device)
        player.cards_hand[batch_indices, self.hand_slot_to_replace, :] = selected_cards

        # Mark the drafted card as unavailable
        selected_deck_indices = torch.gather(indices, 1, index).squeeze(1)  # (batch,)
        self.deck_availability["main"][batch_indices, selected_deck_indices] = False

        # Get probability
        probability = torch.gather(probabilities, 1, index).squeeze(1)  # (batch,)
        return probability

    def _sample_draft_indices(self, deck_type: str, batch_size: int) -> torch.Tensor:
        """Sample draft indices from deck, using sequential dealing if peer_relative_reward.

        Args:
            deck_type: "main" or "bonus"
            batch_size: Number of games in batch

        Returns:
            Tensor of shape (batch, draft_size) with card indices
        """
        if self.peer_relative_reward:
            # Vectorized: all games in same sub-batch see same cards
            # Stack all deck orders: (n_sub_batches, deck_size)
            stacked_orders = torch.stack(self.master_deck_orders[deck_type])

            # Get cursor (same for all sub-batches)
            cursor = self.deck_cursors[deck_type][0]

            # Get indices for each sub-batch: (n_sub_batches, draft_size)
            sub_batch_indices = stacked_orders[:, cursor : cursor + self.draft_size]

            # Compute sub-batch ID for each game: (batch_size,)
            sub_batch_ids = (
                torch.arange(batch_size, device=self.device) // self.current_sub_batch_size
            )

            # Index to get indices for each game: (batch_size, draft_size)
            indices = sub_batch_indices[sub_batch_ids]

            # Update all cursors
            for i in range(self.n_sub_batches):
                self.deck_cursors[deck_type][i] += self.draft_size
        else:
            # Random sampling (original behavior)
            indices = torch.multinomial(
                self.deck_availability[deck_type].float(), self.draft_size, replacement=False
            )
        return indices  # (batch, draft_size)

    def _play_card(self, type: str) -> torch.Tensor:
        """Play a card of the given type (main or bonus).

        For main cards with use_hand=False: samples from deck and plays directly.
        For main cards with use_hand=True: handled by _play_from_hand/_draft_to_hand.
        For bonus cards: always samples from bonus deck.
        """
        batch_size = self.players[0].get_current_batch_size()

        # For bonus cards, check if we need to reshuffle discarded cards
        # Per official rules: "If the Sanctuary deck is empty, shuffle the
        # discarded Sanctuary cards to form a new deck."
        # (Skipped when peer_relative_reward as deck is pre-shuffled and sized appropriately)
        if type == "bonus" and not self.peer_relative_reward:
            self.reshuffle_bonus_discard_if_needed(n_cards_needed=self.draft_size)

        # select indices of cards to sample from the deck
        indices = self._sample_draft_indices(type, batch_size)  # (batch, draft_size)
        # sample the cards
        possible_cards_tensor = self.decks[type][indices]  # (batch, draft_size, MainCard.length())
        probabilities, index, selected_cards = self.players[0].evaluate_cards(
            possible_cards_tensor, self.round_index, mode=type
        )
        if type == "bonus":
            # for playing a bonus card, it depends on the previous main card
            batches_indices_where_card_played = torch.where(
                self.players[0].fields["main"][:, self.round_index, 0]
                > self.players[0].fields["main"][:, self.round_index - 1, 0]
            )[0]
            self.players[0].fields[type][
                batches_indices_where_card_played, self.round_index - 1, :
            ] = selected_cards[batches_indices_where_card_played]
        elif type == "main":
            # all batches play a main card every round
            batches_indices_where_card_played = torch.arange(batch_size, device=self.device)
            self.players[0].play_main_card(selected_cards, self.round_index)

        # update availability tensor
        if self.replace_remaining_cards:
            # only switch the played card to 0
            selected_card_indices = torch.gather(indices, 1, index).squeeze(1)  # (batch,)
            self.deck_availability[type][
                batches_indices_where_card_played,
                selected_card_indices[batches_indices_where_card_played],
            ] = False
        else:
            # flush all draft_size cards
            self.deck_availability[type].scatter_(1, indices, False)

        # Track discarded bonus cards (drawn but not chosen) for reshuffling
        if type == "bonus" and self.replace_remaining_cards:
            # For batches that played a bonus card, track the non-selected cards as discarded
            for batch_idx in batches_indices_where_card_played:
                bid = batch_idx.item()
                selected_idx = index[bid].item()
                # All indices except the selected one go to the discard pile
                batch_indices = indices[bid]  # (draft_size,)
                mask = torch.arange(self.draft_size, device=self.device) != selected_idx
                discarded_indices = batch_indices[mask]
                self.bonus_discard[bid, discarded_indices] = True

        # get the probability of the selected card
        # (1 if no card played, so log(1)=0 won't affect the loss)
        probability = torch.ones(batch_size, device=self.device)
        probability[batches_indices_where_card_played] = torch.gather(
            probabilities, 1, index
        ).squeeze(1)[batches_indices_where_card_played]  # (batch,)
        return probability

    def play_round(self) -> torch.Tensor:
        """Play one round: play a main card, optionally draft, optionally play bonus.

        Returns probabilities of all decisions made this round.
        """
        probabilities_list: list[torch.Tensor] = []

        if self.use_hand:
            # Draft mode: play from hand, then draft to refill hand
            picked_probability_play = self._play_from_hand()
            probabilities_list.append(picked_probability_play)

            # Draft a new card to hand (if cards remain in deck)
            picked_probability_draft = self._draft_to_hand()
            probabilities_list.append(picked_probability_draft)
        else:
            # No draft: pick directly from deck (legacy behavior)
            picked_probability_main = self._play_card("main")
            probabilities_list.append(picked_probability_main)

        # Play bonus card (if applicable)
        if self.use_bonus_cards and self.round_index > 0:
            picked_probability_bonus = self._play_card("bonus")
            probabilities_list.append(picked_probability_bonus)

        # Increment round index
        self.round_index += 1

        # Stack all probabilities
        picked_probabilities = torch.stack(probabilities_list, dim=1)  # (batch, n_decisions)
        return picked_probabilities

    def learning_step(self) -> None:
        batch_size = self.rl_params["train_batch_size"]

        # empty probas tensor for training. dim 0 is batch, but no values
        self.picked_probabilities = torch.ones(batch_size, 0, device=self.device)
        # play the game and get the log probabilities
        self.play_games_batch(batch_size, learning_mode=True)
        log_probs = torch.log(self.picked_probabilities)  # (batch, n_rounds)
        # apply final_count_from_tensor_field for each element of the batch
        scores = self.get_scores()[:, 0]

        # Reward shaping: Count how many times the model played increasing IDs
        # This directly rewards the BEHAVIOR that leads to bonus cards, not the
        # delayed outcome. More direct credit assignment.
        # Normalized to [-1, 1]: 0 increases = -1, 3.5 increases = 0, 7 increases = +1
        id_increase_reward_weight = self.rl_params.get("id_increase_reward_weight", 0.0)
        max_increases = self.n_rounds - 1  # 7 possible increases for 8 rounds
        mid_increases = max_increases / 2.0  # 3.5 = random expectation
        if id_increase_reward_weight > 0:
            player = self.players[0]
            main_card_ids = player.fields["main"][:, :, 0]  # (batch, 8)
            # Count ID increases: id[i+1] > id[i]
            id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]  # (batch, 7) bool
            n_increases = id_diffs.float().sum(dim=1)  # (batch,)
            # Normalize to [-1, 1]
            increases_normalized = (n_increases - mid_increases) / mid_increases
            shaped_reward = scores + id_increase_reward_weight * increases_normalized
        else:
            n_increases = None
            shaped_reward = scores

        # DEPRECATED: bonus_reward_weight - use id_increase_reward_weight instead
        # The bonus card count is an indirect signal; id_increase is the direct behavior
        bonus_reward_weight = self.rl_params.get("bonus_reward_weight", 0.0)
        max_bonus_cards = self.n_rounds - 1  # 7 for 8 rounds
        mid_bonus_cards = max_bonus_cards / 2.0  # 3.5 = random average
        if bonus_reward_weight > 0 and self.use_bonus_cards:
            bonus_cards_played = self.get_bonus_cards_played()[:, 0]  # (batch,)
            bonus_normalized = (bonus_cards_played - mid_bonus_cards) / mid_bonus_cards
            shaped_reward = shaped_reward + bonus_reward_weight * bonus_normalized
        else:
            bonus_cards_played = None

        # Reward shaping: optionally reward playing lower card IDs
        # WARNING: This conflicts with id_increase_reward! Low IDs make increasing harder.
        # Only use this if you specifically need "draft first" behavior over bonus cards.
        # Normalized to [-1, 1]: -1 = ID 68, 0 = ID 34.5 (middle), +1 = ID 1
        low_id_reward_weight = self.rl_params.get("low_id_reward_weight", 0.0)
        if low_id_reward_weight > 0:
            player = self.players[0]
            main_card_ids = player.fields["main"][:, :, 0]  # (batch, 8)
            avg_card_id = main_card_ids.mean(dim=1)  # (batch,)
            max_card_id = 68.0  # Cards are numbered 1-68
            mid_card_id = max_card_id / 2.0  # 34 = middle
            low_id_normalized = (mid_card_id - avg_card_id) / mid_card_id
            shaped_reward = shaped_reward + low_id_reward_weight * low_id_normalized

        # Compute advantage: peer-relative (within-sub-batch normalization) or baseline-based
        if self.peer_relative_reward:
            # Peer-relative: compare against games in the same sub-batch
            # Games in same sub-batch faced identical cards, so differences are purely due
            # to model decisions
            advantage = self._compute_peer_relative_advantage(shaped_reward)
        else:
            # Traditional baseline-based advantage
            advantage = shaped_reward - self.baseline

        loss = (-torch.sum(log_probs, 1) * advantage).mean()  # scalar
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability (especially with mode embedding)
        if self.rl_params.get("grad_clip", None) is not None:
            torch.nn.utils.clip_grad_norm_(
                self.players[0].model.parameters(), self.rl_params["grad_clip"]
            )
        self.optimizer.step()

        # Update total games played (epoch metric based on environment interactions)
        self.players[0].n_training_games_played += self.rl_params["train_batch_size"]
        n_training_games_played = self.players[0].n_training_games_played

        # Log metrics to TensorBoard
        # Using n_training_games_played as x-axis for fair comparison across batch sizes
        if self.writer is not None:
            self.writer.add_scalar(
                "solo_train_score/mean", scores.mean().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/max", scores.max().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/min", scores.min().item(), n_training_games_played
            )
            self.writer.add_scalar(
                "solo_train_score/std", scores.std().item(), n_training_games_played
            )
            # Log bonus cards played (max 7 for 8 rounds)
            if self.use_bonus_cards:
                # Reuse bonus_cards_played if already computed for reward shaping
                if bonus_cards_played is None:
                    bonus_cards_played = self.get_bonus_cards_played()[:, 0]  # (batch,)
                self.writer.add_scalar(
                    "solo_train_bonus/mean",
                    bonus_cards_played.mean().item(),
                    n_training_games_played,
                )
                self.writer.add_scalar(
                    "solo_train_bonus/max", bonus_cards_played.max().item(), n_training_games_played
                )
            # Log average card ID (lower = better for draft priority)
            main_card_ids = self.players[0].fields["main"][:, :, 0]  # (batch, 8)
            self.writer.add_scalar(
                "solo_train_avg_card_id/mean", main_card_ids.mean().item(), n_training_games_played
            )
            # Log ID increases (key metric for bonus card acquisition behavior)
            if n_increases is None:
                id_diffs = main_card_ids[:, 1:] > main_card_ids[:, :-1]
                n_increases = id_diffs.float().sum(dim=1)
            self.writer.add_scalar(
                "solo_train_id_increases/mean", n_increases.mean().item(), n_training_games_played
            )
            # Log shaped reward if using any reward shaping
            if id_increase_reward_weight > 0 or bonus_reward_weight > 0 or low_id_reward_weight > 0:
                self.writer.add_scalar(
                    "solo_train_shaped_reward/mean",
                    shaped_reward.mean().item(),
                    n_training_games_played,
                )
            self.writer.add_scalar("baseline/value", self.baseline, n_training_games_played)
            self.writer.add_scalar(
                "advantage/mean", advantage.mean().item(), n_training_games_played
            )
            self.writer.add_scalar("advantage/std", advantage.std().item(), n_training_games_played)
            self.writer.add_scalar("loss/policy", loss.item(), n_training_games_played)

        if self.verbose > 0:
            logger.info(
                f"Step {self.step_id}. "
                f"Score: {scores.mean().item():.2f}. "
                f"Baseline: {self.baseline:.2f}. "
                f"Loss: {loss.item():.2f}. "
                f"Games: {n_training_games_played}"
            )
        # Update the baseline EMA (use shaped_reward if reward shaping is enabled)
        # In peer_relative_reward mode, baseline is not used for advantage but still
        # tracked for monitoring
        self.baseline = (
            self.baseline
            + self.rl_params["update_baseline_rate"] * (shaped_reward.mean() - self.baseline)
        ).item()
        self.step_id += 1

        # Run periodic evaluations
        if self.eval_vs_random_config and (
            self.step_id % self.eval_vs_random_config.get("every", 500) == 0
            or (self.step_id == 1 and self.eval_vs_random_config.get("initial_eval", False))
        ):
            self.run_eval_vs_random()

        if self.eval_solo_config and (
            self.step_id % self.eval_solo_config.get("every", 500) == 0
            or (self.step_id == 1 and self.eval_solo_config.get("initial_eval", False))
        ):
            self.run_eval_solo()


def main(
    log_to_file: Annotated[bool, typer.Option(help="Whether to log to a file")] = False,
    experiment_name: Annotated[
        str | None, typer.Option(help="Name for TensorBoard experiment")
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 32,
    draft_size: Annotated[int, typer.Option(help="Draft size")] = 10,
    n_steps: Annotated[int, typer.Option(help="Number of training steps")] = 1000,
    n_eval_batches: Annotated[int, typer.Option(help="Number of evaluation batches")] = 100,
    eval_vs_random_every: Annotated[
        int | None, typer.Option(help="Run eval vs random every N steps (None to disable)")
    ] = None,
    eval_vs_random_n_players: Annotated[
        int, typer.Option(help="Number of random players for eval")
    ] = 1,
    eval_solo_every: Annotated[
        int | None, typer.Option(help="Run eval solo every N steps (None to disable)")
    ] = None,
    player_type: Annotated[str, typer.Option(help="Player type: 'mlp' or 'transformer'")] = "mlp",
    use_mode_embedding: Annotated[
        bool, typer.Option(help="Use mode embedding for transformer (play/draft/bonus)")
    ] = False,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 0.0005,
    baseline_update_rate: Annotated[
        float, typer.Option(help="Baseline EMA update rate (lower = smoother)")
    ] = 0.05,
    grad_clip: Annotated[
        float | None, typer.Option(help="Gradient clipping max norm (None to disable)")
    ] = None,
    use_hand: Annotated[
        bool, typer.Option(help="Enable draft mechanism (hand management)")
    ] = False,
    n_cards_hand: Annotated[
        int, typer.Option(help="Number of cards in hand (when using draft)")
    ] = 3,
    peer_relative_reward: Annotated[
        bool, typer.Option(help="Use peer-relative reward (same deck for all games in batch)")
    ] = False,
    advantage_peer_normalization: Annotated[
        str, typer.Option(help="Peer-relative mode: 'zscore' or 'center'")
    ] = "zscore",
) -> None:
    """Run a solo learning game."""
    logger.remove()  # remove default stderr handler
    if log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    eval_vs_random_config: dict[str, Any] | None = None
    if eval_vs_random_every is not None:
        eval_vs_random_config = {
            "every": eval_vs_random_every,
            "n_players": eval_vs_random_n_players,
            "n_batches": n_eval_batches,
            "batch_size": batch_size,
            "initial_eval": True,
        }

    eval_solo_config: dict[str, Any] | None = None
    if eval_solo_every is not None:
        eval_solo_config = {
            "every": eval_solo_every,
            "n_batches": n_eval_batches,
            "batch_size": batch_size,
            "initial_eval": True,
        }

    # Choose model params based on player type
    if player_type == "mlp":
        model_params = {
            "hidden_layers_sizes": [512, 512],
            "dropout_rate": 0.1,
        }
        player_params = {
            "use_cards_hand_in_state": False,
            "use_draft_indicator_in_model_input": False,
        }
    elif player_type == "transformer":
        model_params = {
            "embed_dim": 64,  # 64,
            "n_attention_heads": 4,  # 4,
            "n_transformer_layers": 2,
            "dropout_rate": 0.1,
        }
        player_params = {
            "use_mode_embedding": use_mode_embedding,
        }
    else:
        raise ValueError(f"Unknown player type: {player_type}")

    rl_params: dict[str, Any] = {
        "prior_baseline_score": 29,
        "train_batch_size": batch_size,
        "update_baseline_rate": baseline_update_rate,
        "peer_relative_reward": peer_relative_reward,
        "advantage_peer_normalization": advantage_peer_normalization,
    }
    if grad_clip is not None:
        rl_params["grad_clip"] = grad_clip

    game = SoloLearningGame(
        verbose=2,
        experiment_name=experiment_name,
        model_params=model_params,
        player_params=player_params,
        player_type=player_type,
        optimizer_params={
            "lr": lr,
        },
        rl_params=rl_params,
        draft_size=draft_size,
        use_hand=use_hand,
        n_cards_hand=n_cards_hand,
        eval_vs_random_config=eval_vs_random_config,
        eval_solo_config=eval_solo_config,
    )

    # Log hyperparameters for experiment comparison
    game.log_hparams({"n_steps": n_steps})

    # Training
    for _ in range(n_steps):
        game.learning_step()

    game.dump_player(f"runs/{game.experiment_name}/player.pt")
    game.close_tensorboard()
    print(f"\nTensorBoard logs saved to: runs/{game.experiment_name}")
    print("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

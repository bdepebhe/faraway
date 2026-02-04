"""
Real batched games using nn bots.
"""

import sys
from collections.abc import Sequence
from typing import Annotated

import torch
import torch as torch_module
import typer
from loguru import logger

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import MainCard, MainCardsSeries
from faraway.core.human_player import HumanPlayer
from faraway.torch.base_game import BaseNNGame
from faraway.torch.mlp_player import MLPPlayer
from faraway.torch.nn_player import BaseNNPlayer
from faraway.torch.transformers_player import TransformersPlayer

MAP_INDEX_IN_FLATTENED_CARD = MainCard.get_field_index("map", "assets")


class RealNNGame(BaseNNGame):
    def __init__(
        self,
        players: Sequence[BasePlayer],
        n_rounds: int = 8,
        use_bonus_cards: bool = True,
        device: torch.device | None = None,
        verbose: int = 0,
        experiment_name: str | None = None,
        log_dir: str = "runs",
    ):
        super().__init__(
            n_rounds, use_bonus_cards, device, players, verbose, experiment_name, log_dir
        )
        self.verbose = verbose
        self.draft_pool: torch.Tensor | None = None

    def reset_games_batch(self, batch_size: int) -> None:
        # reset the deck and the fields
        super().reset_games_batch(batch_size)
        # deal initial hands to all players (3 cards each)
        self.deal_initial_hands(n_cards=3, batch_size=batch_size)
        # Track draft priority wins per player: (batch_size, n_players)
        # Counts how many times each player had the lowest card ID (drafts first)
        self.draft_priority_wins = torch.zeros(
            batch_size, len(self.players), dtype=torch.long, device=self.device
        )
        if self.verbose > 98:
            logger.debug(f"BATCH SETUP: batch_size={batch_size}")
            for i in range(batch_size):
                logger.debug(f"Game #{i}")
                used_cards = [
                    1 + card_id
                    for card_id in torch.where(~self.deck_availability["main"][i])[0].tolist()
                ]
                logger.debug(f"    Used cards: {used_cards}")
                for p, player in enumerate(self.players):
                    logger.debug(
                        f"    Player #{p} receives cards hand: "
                        f"{MainCardsSeries.from_numpy(player.cards_hand[i])}"
                    )

    def draw_draft_pool(self) -> torch.Tensor:
        batch_size = self.deck_availability["main"].shape[0]
        card_length = self.decks["main"].shape[1]
        # expand the main deck to batch size: (68, 24) -> (batch_size, 68, 24)
        expanded_main_deck = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)
        # create common draft pool - indices shape: (batch_size, n_players+1)
        indices = torch.multinomial(
            self.deck_availability["main"].float(), len(self.players) + 1, replacement=False
        )
        # expand indices for gather: (batch_size, n_players+1) -> (batch_size, n_players+1, 24)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
        # gather cards: (batch_size, 68, 24) -> (batch_size, n_players+1, 24)
        draft_pool = torch.gather(expanded_main_deck, 1, indices_expanded)
        # update the deck availability
        self.deck_availability["main"].scatter_(1, indices, False)
        if self.verbose > 98:
            logger.debug(f"DRAW DRAFT POOL: batch_size={batch_size}")
            for i in range(batch_size):
                logger.debug(f"Game #{i}")
                logger.debug(f"    Used cards: {self.get_used_cards_ids('main', i)}")
                logger.debug(f"    Draft pool: {MainCardsSeries.from_numpy(draft_pool[i])}")
        return draft_pool

    def play_round(self) -> None:
        batch_size = self.players[0].get_current_batch_size()
        # each player plays a main card
        index_played_from_hand = torch.zeros(
            batch_size, len(self.players), dtype=torch.long, device=self.device
        )
        n_bonus_cards_to_draw = torch.zeros(
            batch_size, len(self.players), dtype=torch.long, device=self.device
        )
        if self.verbose > 98:
            logger.debug(
                f"################### PLAY OF MAIN CARD ROUND #{self.round_index}: "
                f"batch_size={batch_size}"
            )
        for p, player in enumerate(self.players):
            # evaluate the cards
            _, index, selected_cards = player.evaluate_cards(
                player.cards_hand, self.round_index, mode="play"
            )
            player.play_main_card(selected_cards, self.round_index)
            # keep track of the index of the card played from the hand,
            # so we can place next card in the same position
            index_played_from_hand[:, p] = torch.tensor(index, device=self.device).squeeze()
            n_maps_in_main_cards = (
                torch.tensor(player.fields["main"][:, :, MAP_INDEX_IN_FLATTENED_CARD])
                .sum(dim=1)
                .long()
            )
            n_maps_in_bonus_cards = (
                torch.tensor(player.fields["bonus"][:, :, MAP_INDEX_IN_FLATTENED_CARD])
                .sum(dim=1)
                .long()
            )
            n_bonus_cards_to_draw[:, p] = n_maps_in_main_cards + n_maps_in_bonus_cards + 1
            if self.verbose > 98:
                logger.debug(f"Player #{p}")
                for i in range(batch_size):
                    logger.debug(f"    Game #{i}")
                    formatted_probas = ", ".join(f"{p:.2f}" for p in _[i].tolist())
                    logger.debug(
                        f"    Player #{p} evaluated probas: [{formatted_probas}] "
                        f"and selected #{index[i].item()}"
                    )
                    logger.debug(
                        f"    Player #{p} plays card: {MainCard.from_numpy(selected_cards[i])}"
                    )

        # We compare the id of last card played, this will be used to determine,
        # for each element of the batch, the order in which the players do the following actions
        # Shape: (batch_size, n_players) - each row has the card IDs played by each player
        # for that batch element
        last_card_played_ids = torch.stack(
            [
                torch.tensor(player.fields["main"], device=self.device)[:, self.round_index, 0]
                for player in self.players
            ],
            dim=1,
        )

        # Track draft priority: who played the lowest card ID (gets to draft first)
        # Only count for rounds 0-6 (7 draft opportunities, no draft in round 7)
        if self.round_index < self.n_rounds - 1:
            draft_winner = last_card_played_ids.argmin(dim=1)  # (batch_size,)
            # One-hot encode and accumulate
            self.draft_priority_wins.scatter_add_(
                1,
                draft_winner.unsqueeze(1),
                torch.ones(batch_size, 1, dtype=torch.long, device=self.device),
            )

        draft_pool = self.draw_draft_pool()

        # Batched draft resolution (one forward pass per player instead of per player per game)
        self.resolve_draft_batched(
            last_card_played_ids,
            draft_pool,
            index_played_from_hand,
        )

        # Batched bonus card resolution (one forward pass per player)
        if self.verbose > 98:
            logger.debug("RESOLVE BONUS CARDS (BATCHED)")
        self.resolve_bonus_cards_batched(n_bonus_cards_to_draw)

        self.round_index += 1

    def resolve_draft_batched(
        self,
        last_card_played_ids: torch.Tensor,  # (batch_size, n_players)
        draft_pool: torch.Tensor,  # (batch_size, n_players + 1, card_length)
        index_played_from_hand: torch.Tensor,  # (batch_size, n_players)
    ) -> None:
        """Resolve draft phase for all games in a batched manner.

        Key optimization: Pre-compute all player evaluations in one forward pass per player,
        then allocate cards sequentially using masking (no additional forward passes).
        """
        if self.round_index >= self.n_rounds - 1:
            return  # No draft in last round

        batch_size = draft_pool.shape[0]
        n_draft_cards = draft_pool.shape[1]
        n_players = len(self.players)

        # Pre-compute logits for all players on full draft pool (one forward pass per player)
        # Shape: (n_players, batch_size, n_draft_cards)
        all_logits_list = []
        for player in self.players:
            logits, _, _ = player.evaluate_cards(
                draft_pool, self.round_index, mode="draft", return_logits=True
            )
            all_logits_list.append(logits)
        all_logits = torch.stack(all_logits_list, dim=0)  # (n_players, batch_size, n_draft_cards)

        # Track which cards are taken: (batch_size, n_draft_cards)
        card_taken = torch.zeros(batch_size, n_draft_cards, dtype=torch.bool, device=self.device)

        # Determine priority order for each game (lowest card ID goes first)
        # priority_order[game, rank] = player_idx
        priority_order = last_card_played_ids.argsort(dim=1)  # (batch_size, n_players)

        # Allocate cards in priority order
        for rank in range(n_players):
            # Get which player has this priority rank for each game
            player_indices = priority_order[:, rank]  # (batch_size,)

            # Gather logits for the appropriate player for each game
            # all_logits: (n_players, batch_size, n_draft_cards)
            # We need logits[player_indices[b], b, :] for each b
            batch_indices = torch.arange(batch_size, device=self.device)
            current_logits = all_logits[
                player_indices, batch_indices, :
            ]  # (batch_size, n_draft_cards)

            # Mask out already-taken cards
            current_logits = current_logits.masked_fill(card_taken, float("-inf"))

            # Select best card for each game
            selected_indices = current_logits.argmax(dim=1)  # (batch_size,)

            # Mark cards as taken
            card_taken.scatter_(1, selected_indices.unsqueeze(1), True)

            # Get the selected cards
            card_length = draft_pool.shape[2]
            selected_indices_expanded = (
                selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, card_length)
            )
            selected_cards = torch.gather(draft_pool, 1, selected_indices_expanded).squeeze(
                1
            )  # (batch_size, card_length)

            # Place cards in each player's hand at the correct position (vectorized per player)
            batch_indices = torch.arange(batch_size, device=self.device)
            hand_positions = index_played_from_hand[batch_indices, player_indices]  # (batch_size,)
            for p in range(n_players):
                mask = player_indices == p
                if mask.any():
                    batch_p = mask.nonzero(as_tuple=True)[0]
                    hand_p = index_played_from_hand[batch_p, p]
                    self.players[p].cards_hand[batch_p, hand_p, :] = selected_cards[batch_p]

            if self.verbose > 98:
                for game_idx in range(batch_size):
                    player_idx = player_indices[game_idx].item()
                    hand_position = hand_positions[game_idx].item()
                    logger.debug(
                        f"    Game #{game_idx}: Player #{player_idx} drafts card "
                        f"{MainCard.from_numpy(selected_cards[game_idx].cpu().numpy())} "
                        f"at hand position {hand_position}"
                    )

    def resolve_bonus_cards_batched(
        self,
        n_bonus_cards_to_draw: torch.Tensor,  # (batch_size, n_players)
    ) -> None:
        """Resolve bonus card draws in batched manner (one forward pass per player).

        Process each player's bonus draws across all games in a single batch.
        Order between players doesn't matter for evaluation purposes.
        """
        if self.round_index == 0:
            return  # No bonus cards in first round

        card_length = self.decks["bonus"].shape[1]

        for player_idx, player in enumerate(self.players):
            # Find games where this player triggers bonus (current > previous)
            triggers_bonus = (
                player.fields["main"][:, self.round_index, 0]
                > player.fields["main"][:, self.round_index - 1, 0]
            )  # (batch_size,)

            if not triggers_bonus.any():
                continue

            game_indices = triggers_bonus.nonzero().squeeze(1)  # indices of games to process
            n_games = len(game_indices)

            # Get number of cards to draw per game
            n_cards_per_game = n_bonus_cards_to_draw[game_indices, player_idx]  # (n_games,)
            max_cards = int(n_cards_per_game.max().item())

            if max_cards == 0:
                continue

            # Batch-draw bonus cards for all triggered games
            # We'll pad to max_cards and use masking
            all_bonus_cards = torch.zeros(n_games, max_cards, card_length, device=self.device)
            valid_mask = torch.zeros(n_games, max_cards, dtype=torch.bool, device=self.device)
            drawn_indices_padded = torch.zeros(
                n_games, max_cards, dtype=torch.long, device=self.device
            )

            for i, game_idx in enumerate(game_indices):
                gid = game_idx.item()
                n_to_draw = int(n_cards_per_game[i].item())

                if n_to_draw == 0:
                    continue

                # Reshuffle if needed
                self.reshuffle_bonus_discard_if_needed(
                    game_ids=torch.tensor([gid], device=self.device),
                    n_cards_needed=n_to_draw,
                )

                # Draw cards
                n_available = self.deck_availability["bonus"][gid].sum().item()
                if n_to_draw > n_available:
                    raise ValueError(
                        f"Game {gid}: Can't draw {n_to_draw} bonus cards, "
                        f"only {n_available} available"
                    )

                indices = torch.multinomial(
                    self.deck_availability["bonus"][gid : gid + 1].float(),
                    n_to_draw,
                    replacement=False,
                ).squeeze(0)  # (n_to_draw,)

                # Mark as unavailable
                self.deck_availability["bonus"][gid].scatter_(0, indices, False)

                # Gather the cards
                drawn_cards = self.decks["bonus"][indices]  # (n_to_draw, card_length)

                # Store in padded tensors
                all_bonus_cards[i, :n_to_draw, :] = drawn_cards
                valid_mask[i, :n_to_draw] = True
                drawn_indices_padded[i, :n_to_draw] = indices

            # Single forward pass for all games where this player draws
            logits, _, _ = player.evaluate_cards(
                all_bonus_cards,
                self.round_index,
                mode="bonus",
                games_indices=game_indices,
                return_logits=True,
            )  # (n_games, max_cards)

            # Mask invalid (padded) positions
            logits = logits.masked_fill(~valid_mask, float("-inf"))

            # Select best card per game
            selected_indices = logits.argmax(dim=1)  # (n_games,)

            # Gather selected cards and assign to player fields (vectorized)
            selected_indices_expanded = (
                selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, card_length)
            )
            selected_cards = torch.gather(all_bonus_cards, 1, selected_indices_expanded).squeeze(1)
            player.fields["bonus"][game_indices, self.round_index - 1, :] = selected_cards

            # Track discarded cards (drawn but not selected) - vectorized
            discarded_mask = valid_mask & (
                torch.arange(max_cards, device=self.device).unsqueeze(0)
                != selected_indices.unsqueeze(1)
            )
            if discarded_mask.any():
                row_idx = game_indices.unsqueeze(1).expand(-1, max_cards)[discarded_mask]
                col_idx = drawn_indices_padded[discarded_mask]
                self.bonus_discard[row_idx, col_idx] = True

            if self.verbose > 98:
                for i, gid in enumerate(game_indices.tolist()):
                    logger.debug(
                        f"    Game #{gid}: Player #{player_idx} plays bonus card "
                        f"{MainCard.from_numpy(selected_cards[i].cpu().numpy())}"
                    )

    def get_draft_priority_rate(self) -> torch.Tensor:
        """Get the draft priority win rate for each player.

        Returns:
            Tensor of shape (batch_size, n_players) with rate in [0, 1].
            Each value is the fraction of 7 draft rounds where that player had priority.
        """
        n_draft_rounds = self.n_rounds - 1  # 7 draft opportunities
        return self.draft_priority_wins.float() / n_draft_rounds

    def run_tournament(
        self,
        n_batches: int,
        batch_size: int,
        player_names: list[str] | None = None,
    ) -> tuple[list[int], list[float], list[float], list[float]]:
        """Run a tournament and return stats per player.

        Args:
            n_batches: Number of batches to play
            batch_size: Number of games per batch
            player_names: Optional names for TensorBoard logging

        Returns:
            Tuple of (wins, mean_scores, mean_bonus_cards, mean_draft_priority_rate) per player
        """
        all_scores: list[torch.Tensor] = []
        all_bonus_cards: list[torch.Tensor] = []
        all_draft_priority: list[torch.Tensor] = []

        for i in range(n_batches):
            self.play_games_batch(batch_size)
            if self.verbose > 1:
                logger.info(f"Batch {i + 1} completed")
            all_scores.append(self.get_scores())
            if self.use_bonus_cards:
                all_bonus_cards.append(self.get_bonus_cards_played())
            all_draft_priority.append(self.get_draft_priority_rate())

        scores = torch.cat(all_scores, dim=0)  # (total_games, players)
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
            bonus_cards = torch.cat(all_bonus_cards, dim=0)
            mean_bonus_cards = bonus_cards.mean(dim=0).tolist()
        else:
            mean_bonus_cards = [0.0] * len(self.players)

        # Compute mean draft priority rate per player
        draft_priority = torch.cat(all_draft_priority, dim=0)  # (total_games, players)
        mean_draft_priority = draft_priority.mean(dim=0).tolist()

        if self.verbose > 0:
            logger.info(
                f"Tournament completed.\n"
                f"Mean scores: {mean_scores}\n"
                f"Wins: {wins}\n"
                f"Win rate: {win_rate}%\n"
                f"Mean bonus cards: {mean_bonus_cards}\n"
                f"Mean draft priority rate: {[f'{r:.1%}' for r in mean_draft_priority]}\n"
            )

        return wins, mean_scores, mean_bonus_cards, mean_draft_priority

    def resolve_actions_one_game(
        self,
        last_card_played_ids: torch.Tensor,  # (n_players,)
        draft_pool: torch.Tensor,  # (n_players + 1, card_length)
        index_played_from_hand: torch.Tensor,  # (n_players,)
        n_bonus_cards_to_draw: torch.Tensor,  # (n_players,)
        game_id: int,
    ) -> None:
        """Legacy method - kept for backwards compatibility."""
        # exapnad( add one dim before other dims)
        draft_pool = draft_pool[:, :, :].clone()
        # loop other players from highest to lowest card id
        while last_card_played_ids.min() < 100:
            # find the player with the lowest card id
            p = last_card_played_ids.argmin()
            if self.verbose > 98:
                logger.debug(
                    f"    Player #{p} is next to play with card id: "
                    f"{int(last_card_played_ids[p].item())}"
                )
            last_card_played_ids[p] = 100
            draft_pool = self.resolve_actions_one_player(
                self.players[p],
                draft_pool,
                index_played_from_hand[p].item(),
                n_bonus_cards_to_draw[p].item(),
                game_id=game_id,
            )

    def resolve_actions_one_player(
        self,
        player: BaseNNPlayer,
        draft_pool: torch.Tensor,
        index_played_from_hand: int,
        n_bonus_cards_to_draw: int,
        game_id: int,
    ) -> torch.Tensor:
        if self.round_index < self.n_rounds - 1:  # no need to draw draft cards in the last round
            # evaluate the cards of the common draft pool
            _, index, selected_card = player.evaluate_cards(
                draft_pool,
                self.round_index,
                mode="draft",
                games_indices=slice(game_id, game_id + 1),
            )
            # update the draft pool: remove the selected card from the tensor
            selected_index = int(index.squeeze()) if hasattr(index, "squeeze") else int(index)
            mask = torch.arange(draft_pool.shape[1], device=self.device) != selected_index
            draft_pool = draft_pool[:, mask, :]
            # place card in the player's hand where the previously played card was
            player.cards_hand[game_id, index_played_from_hand, :] = torch.tensor(
                selected_card, device=self.device
            ).squeeze(0)

            if self.verbose > 98:
                formatted_probas = ", ".join(f"{p:.2f}" for p in _[0].tolist())
                logger.debug(
                    f"    Player evaluated probas: [{formatted_probas}] "
                    f"and selected #{index[0].item()}"
                )
                logger.debug(f"    Player selected card: {MainCard.from_numpy(selected_card[0])}")
                logger.debug(f"    Player places card in hand at position {index_played_from_hand}")
                logger.debug(
                    f"    Player has now cards hand: "
                    f"{MainCardsSeries.from_numpy(player.cards_hand[game_id, :, :])}"
                )
                logger.debug(
                    f"    Player has main field: "
                    f"{MainCardsSeries.from_numpy(player.fields['main'][game_id, :, :])}"
                )

        if self.round_index > 0:  # no bonus cards in the first round
            # check if the previously played card is lower than the current card
            if (
                player.fields["main"][game_id, self.round_index, 0]
                > player.fields["main"][game_id, self.round_index - 1, 0]
            ):
                # Check if we need to reshuffle discarded bonus cards
                # Per official rules: "If the Sanctuary deck is empty, shuffle the
                # discarded Sanctuary cards to form a new deck."
                self.reshuffle_bonus_discard_if_needed(
                    game_ids=torch.tensor([game_id], device=self.device),
                    n_cards_needed=n_bonus_cards_to_draw,
                )

                # cap to available bonus cards (after potential reshuffle)
                n_available = self.deck_availability["bonus"][game_id, :].sum().item()
                if n_bonus_cards_to_draw > n_available:
                    raise ValueError(
                        f"Player {player} can't draw {n_bonus_cards_to_draw} "
                        f"bonus cards, only {n_available} available (even after reshuffle)"
                    )

                indices = torch.multinomial(
                    self.deck_availability["bonus"].float()[game_id : game_id + 1, :],
                    n_bonus_cards_to_draw,
                    replacement=False,
                )
                self.deck_availability["bonus"][game_id : game_id + 1, :].scatter_(
                    1, indices, False
                )

                # gather bonus cards: expand indices and use gather
                card_length = self.decks["bonus"].shape[1]
                indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
                expanded_bonus_deck = self.decks["bonus"].unsqueeze(0)  # (1, 45, 24)
                bonus_cards_drawn = torch.gather(
                    expanded_bonus_deck, 1, indices_expanded
                )  # (1, n_to_draw, 24)

                # evaluate the draw of bonus cards
                _, index, selected_card = player.evaluate_cards(
                    bonus_cards_drawn,
                    self.round_index,
                    mode="bonus",
                    games_indices=slice(game_id, game_id + 1),
                )
                player.fields["bonus"][game_id : game_id + 1, self.round_index - 1, :] = (
                    selected_card
                )

                # Track discarded bonus cards (drawn but not chosen) for reshuffling
                # The selected card stays "used", the others go to the discard pile
                selected_index = int(index.squeeze()) if hasattr(index, "squeeze") else int(index)
                discarded_indices = indices.squeeze(0).clone()
                # Create mask to exclude selected card
                mask = (
                    torch.arange(discarded_indices.shape[0], device=self.device) != selected_index
                )
                discarded_indices = discarded_indices[mask]  # All indices except the selected one
                # Add discarded cards to the discard pile
                if discarded_indices.numel() > 0:
                    self.bonus_discard[game_id, discarded_indices] = True
                if self.verbose > 98:
                    logger.debug(f"    Player draws {n_bonus_cards_to_draw} bonus cards")
                    logger.debug(
                        f"    Player draws bonus cards: "
                        f"{MainCardsSeries.from_numpy(bonus_cards_drawn[0])}"
                    )
                    formatted_probas = ", ".join(f"{p:.2f}" for p in _[0].tolist())
                    logger.debug(
                        f"    Player evaluated bonus probas: [{formatted_probas}] "
                        f"and selected #{index[0].item()}"
                    )
                    logger.debug(
                        f"    Player plays bonus card: " f"{MainCard.from_numpy(selected_card[0])}"
                    )
                    logger.debug(
                        f"    Player has bonus field: "
                        f"{MainCardsSeries.from_numpy(player.fields['bonus'][game_id, :, :])}"
                    )
                    logger.debug(
                        f"    Used bonus cards: " f"{self.get_used_cards_ids('bonus', game_id)}"
                    )
        return draft_pool


def main(
    players: Annotated[
        list[str],
        typer.Argument(help="Paths to the players to use (e.g., model.pt or 'human' or 'random')"),
    ],
    log_to_file: Annotated[bool, typer.Option(help="Whether to log to a file")] = False,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 32,
    n_batches: Annotated[int, typer.Option(help="Number of batches to play")] = 100,
    verbose: Annotated[int, typer.Option(help="Verbosity level")] = 1,
    experiment_name: Annotated[
        str | None, typer.Option(help="Name for TensorBoard experiment (enables logging)")
    ] = None,
    log_dir: Annotated[str, typer.Option(help="Directory for TensorBoard logs")] = "runs",
) -> None:
    """Run a tournament between NN players."""
    logger.remove()  # remove default stderr handler
    if log_to_file:
        logger.add(f"{log_dir}/{experiment_name}/real_game.log")
    else:
        logger.add(sys.stdout)
    n_rounds = 8

    # load the players
    players_list: list[BasePlayer] = []
    player_names: list[str] = []
    for player in players or []:
        if player.endswith(".pt"):
            # Auto-detect player type from checkpoint
            checkpoint = torch_module.load(player, map_location="cpu", weights_only=False)
            player_type = checkpoint.get(
                "player_type", "mlp"
            )  # default to mlp for backwards compat
            if player_type == "transformer":
                players_list.append(TransformersPlayer.load(player))
            else:
                players_list.append(MLPPlayer.load(player))
            # Extract a short name from the path for TensorBoard labels
            player_names.append(player.replace("/", "_").replace(".pt", ""))
        elif player == "human":
            players_list.append(HumanPlayer(n_rounds))
            player_names.append("human")
        else:
            raise ValueError(f"Unknown player type: {player}")
        if players_list[-1].n_rounds != n_rounds:
            raise ValueError(
                f"Player {players_list[-1]} has {players_list[-1].n_rounds} rounds, but "
                f"current game has {n_rounds} rounds"
            )
    game = RealNNGame(
        players=players_list,
        n_rounds=n_rounds,
        verbose=verbose,
        experiment_name=experiment_name,
        log_dir=log_dir,
    )
    logger.info(f"Starting real games tournament with {len(players_list)} players")
    logger.info(f"Players: {player_names}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of batches: {n_batches}")
    logger.info(f"Verbose: {verbose}")
    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Log directory: {log_dir}")

    # Initialize TensorBoard if experiment name is provided
    if experiment_name is not None:
        game.init_tensorboard(default_prefix="eval")
        logger.info(f"TensorBoard logging enabled: {log_dir}/{game.experiment_name}")

    game.run_tournament(n_batches=n_batches, batch_size=batch_size, player_names=player_names)

    # Close TensorBoard writer
    if game.writer is not None:
        game.close_tensorboard()
        logger.info(f"TensorBoard logs saved to: {log_dir}/{game.experiment_name}")
        logger.info("Run 'tensorboard --logdir=runs' to view results")


if __name__ == "__main__":
    typer.run(main)

"""
Batched Faraway game environment.

Pure game state and transitions: no RL, no optimizer. Supports solo (n_players=1)
and multiplayer (n_players>=2). Used by learning runners and evaluation.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from faraway.torch.base_game import (
    final_count_tensor_batched,
    get_bonus_deck_tensor,
    get_main_deck_tensor,
)


@dataclass
class RoundResult:
    """Result of one round for the runner (e.g. to accumulate log-probs)."""

    # Per-player decision probabilities this round. Solo: list of 1 tensor (batch, n_decisions).
    # Multiplayer: list of n_players tensors (batch, n_decisions) or empty if not tracked.
    picked_probabilities: list[torch.Tensor]
    # Multiplayer only: (batch_size,) player index who had draft priority this round, or None.
    draft_winner_this_round: torch.Tensor | None = None


class PlayerLike(Protocol):
    """Protocol for objects that can act as players in the env (e.g. BaseNNPlayer)."""

    fields: dict[str, torch.Tensor]
    cards_hand: torch.Tensor

    def reset_games_batch(self, batch_size: int) -> None: ...
    def get_current_batch_size(self) -> int: ...
    def play_main_card(self, selected_cards: torch.Tensor, round_index: int) -> None: ...

    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: torch.Tensor | slice | None = None,
        return_logits: bool = False,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


# Map asset index in flattened card (position 0 = id, 1-9 = assets; map = 8)
MAP_INDEX_IN_FLATTENED_CARD = 1 + 8  # IDX_ASSETS_START + ASSET_MAP


class BatchedFarawayEnv:
    """
    Batched Faraway game environment. Holds deck state and round index; mutates
    player state (fields, cards_hand) via step_round(players).

    Supports:
    - Solo (len(players)==1): play main from hand or draft, optional draft to hand, optional bonus.
    - Multiplayer (len(players)>=2): each player plays main, draft pool by priority, then bonus.
    """

    def __init__(
        self,
        n_rounds: int = 8,
        use_bonus_cards: bool = True,
        draft_size: int = 10,
        use_hand: bool = False,
        n_cards_hand: int = 3,
        replace_remaining_cards: bool = True,
        device: torch.device | None = None,
        peer_relative_reward: bool = False,
        peer_sub_batch_size: int | None = None,
        verbose: int = 0,
    ):
        self.n_rounds = n_rounds
        self.use_bonus_cards = use_bonus_cards
        self.draft_size = draft_size
        self.use_hand = use_hand
        self.n_cards_hand = n_cards_hand
        self.replace_remaining_cards = replace_remaining_cards
        self.peer_relative_reward = peer_relative_reward
        self.peer_sub_batch_size = peer_sub_batch_size
        self.verbose = verbose

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        main_deck = get_main_deck_tensor().to(self.device)
        bonus_deck = get_bonus_deck_tensor().to(self.device)
        self.decks = {"main": main_deck, "bonus": bonus_deck}

        self.deck_availability: dict[str, torch.Tensor]
        self.bonus_discard: torch.Tensor
        self.round_index: int = 0

        # Peer-relative: fixed deck order per sub-batch
        self.master_deck_orders: dict[str, list[torch.Tensor]]
        self.deck_cursors: dict[str, list[int]]
        self.n_sub_batches: int = 0
        self.current_sub_batch_size: int = 0
        self.hand_slot_to_replace: torch.Tensor | None = None

    def reset(self, batch_size: int, players: Sequence[PlayerLike]) -> None:
        """Reset deck state and all players for a new batch of games."""
        self.deck_availability = {
            "main": torch.ones(
                batch_size, self.decks["main"].shape[0], dtype=torch.bool, device=self.device
            ),
            "bonus": torch.ones(
                batch_size, self.decks["bonus"].shape[0], dtype=torch.bool, device=self.device
            ),
        }
        self.bonus_discard = torch.zeros(
            batch_size, self.decks["bonus"].shape[0], dtype=torch.bool, device=self.device
        )
        self.round_index = 0
        self.hand_slot_to_replace = None

        for p in players:
            p.reset_games_batch(batch_size)

        if self.peer_relative_reward:
            self._setup_fixed_seed_decks(batch_size)
            if self.use_hand and len(players) == 1:
                self._deal_initial_hands_fixed_seed(players[0], batch_size)
        elif self.use_hand or (len(players) > 1):
            n_cards = self.n_cards_hand if len(players) == 1 else 3
            self._deal_initial_hands(players, batch_size, n_cards=n_cards)

    def _setup_fixed_seed_decks(self, batch_size: int) -> None:
        sub_batch_size = self.peer_sub_batch_size or batch_size
        self.n_sub_batches = (batch_size + sub_batch_size - 1) // sub_batch_size
        self.current_sub_batch_size = sub_batch_size
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
        self.deck_cursors = {"main": [0] * self.n_sub_batches, "bonus": [0] * self.n_sub_batches}

    def _deal_initial_hands(
        self, players: Sequence[PlayerLike], batch_size: int, n_cards: int | None = None
    ) -> None:
        n_cards = n_cards or self.n_cards_hand
        card_length = self.decks["main"].shape[1]
        expanded_main = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)
        for player in players:
            indices = torch.multinomial(
                self.deck_availability["main"].float(), n_cards, replacement=False
            )
            indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
            player.cards_hand = torch.gather(expanded_main, 1, indices_expanded)
            self.deck_availability["main"].scatter_(1, indices, False)

    def _deal_initial_hands_fixed_seed(self, player: PlayerLike, batch_size: int) -> None:
        n_cards = self.n_cards_hand
        card_length = self.decks["main"].shape[1]
        expanded_main = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)
        stacked = torch.stack(self.master_deck_orders["main"])
        cursor = self.deck_cursors["main"][0]
        sub_batch_indices = stacked[:, cursor : cursor + n_cards]
        sub_batch_ids = torch.arange(batch_size, device=self.device) // self.current_sub_batch_size
        indices = sub_batch_indices[sub_batch_ids]
        for i in range(self.n_sub_batches):
            self.deck_cursors["main"][i] += n_cards
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
        player.cards_hand = torch.gather(expanded_main, 1, indices_expanded)
        self.deck_availability["main"].scatter_(1, indices, False)

    def _sample_draft_indices(self, deck_type: str, batch_size: int) -> torch.Tensor:
        if self.peer_relative_reward:
            stacked = torch.stack(self.master_deck_orders[deck_type])
            cursor = self.deck_cursors[deck_type][0]
            sub_batch_indices = stacked[:, cursor : cursor + self.draft_size]
            sub_batch_ids = (
                torch.arange(batch_size, device=self.device) // self.current_sub_batch_size
            )
            indices = sub_batch_indices[sub_batch_ids]
            for i in range(self.n_sub_batches):
                self.deck_cursors[deck_type][i] += self.draft_size
        else:
            indices = torch.multinomial(
                self.deck_availability[deck_type].float(), self.draft_size, replacement=False
            )
        return indices

    def reshuffle_bonus_discard_if_needed(
        self, game_ids: torch.Tensor | None = None, n_cards_needed: int = 1
    ) -> None:
        if game_ids is None:
            game_ids = torch.arange(self.deck_availability["bonus"].shape[0], device=self.device)
        for game_id in game_ids:
            gid = int(game_id.item()) if isinstance(game_id, torch.Tensor) else game_id
            n_available = self.deck_availability["bonus"][gid].sum().item()
            if n_available < n_cards_needed:
                n_discarded = self.bonus_discard[gid].sum().item()
                if n_discarded > 0:
                    self.deck_availability["bonus"][gid] |= self.bonus_discard[gid]
                    self.bonus_discard[gid] = False

    def get_scores(self, players: Sequence[PlayerLike]) -> torch.Tensor:
        """Return (batch, n_players) final scores."""
        return torch.stack(
            [final_count_tensor_batched(p.fields["main"], p.fields["bonus"]) for p in players],
            dim=1,
        )

    def get_bonus_cards_played(self, players: Sequence[PlayerLike]) -> torch.Tensor:
        """Return (batch, n_players) count of bonus cards played."""
        return torch.stack(
            [(p.fields["bonus"][:, :, 0] > 0).sum(dim=1).float() for p in players],
            dim=1,
        )

    def step_round(self, players: Sequence[PlayerLike], temperature: float = 1.0) -> RoundResult:
        """Play one round; mutate players' fields and cards_hand. Return round result for runner."""
        if len(players) == 1:
            return self._step_round_solo(players[0], temperature)
        return self._step_round_multiplayer(players)

    def _step_round_solo(self, player: PlayerLike, temperature: float) -> RoundResult:
        probs_list: list[torch.Tensor] = []
        batch_size = player.get_current_batch_size()

        if self.use_hand:
            # Play from hand
            probabilities, index, selected_cards = player.evaluate_cards(
                player.cards_hand, self.round_index, mode="play", temperature=temperature
            )
            player.play_main_card(selected_cards, self.round_index)
            self.hand_slot_to_replace = index.squeeze(1)
            probs_list.append(torch.gather(probabilities, 1, index).squeeze(1))

            # Draft to hand
            if not self.peer_relative_reward:
                if self.deck_availability["main"].sum(dim=1).min() < self.draft_size:
                    probs_list.append(torch.ones(batch_size, device=self.device))
                    if self.use_bonus_cards and self.round_index > 0:
                        probs_list.append(self._solo_play_bonus(player, temperature))
                    self.round_index += 1
                    return RoundResult(picked_probabilities=[torch.stack(probs_list, dim=1)])
            indices = self._sample_draft_indices("main", batch_size)
            river_cards = self.decks["main"][indices]
            probabilities, index, selected_cards = player.evaluate_cards(
                river_cards, self.round_index, mode="draft", temperature=temperature
            )
            batch_indices = torch.arange(batch_size, device=self.device)
            player.cards_hand[batch_indices, self.hand_slot_to_replace, :] = selected_cards
            selected_deck_idx = torch.gather(indices, 1, index).squeeze(1)
            self.deck_availability["main"][batch_indices, selected_deck_idx] = False
            probs_list.append(torch.gather(probabilities, 1, index).squeeze(1))
        else:
            # Play main from deck
            probs_list.append(self._solo_play_card(player, "main", batch_size, temperature))

        if self.use_bonus_cards and self.round_index > 0:
            probs_list.append(self._solo_play_bonus(player, temperature))

        self.round_index += 1
        return RoundResult(picked_probabilities=[torch.stack(probs_list, dim=1)])

    def _solo_play_card(
        self, player: PlayerLike, card_type: str, batch_size: int, temperature: float
    ) -> torch.Tensor:
        if card_type == "bonus" and not self.peer_relative_reward:
            self.reshuffle_bonus_discard_if_needed(n_cards_needed=self.draft_size)
        indices = self._sample_draft_indices(card_type, batch_size)
        possible_cards = self.decks[card_type][indices]
        probabilities, index, selected_cards = player.evaluate_cards(
            possible_cards, self.round_index, mode=card_type, temperature=temperature
        )
        if card_type == "bonus":
            batches_played = torch.where(
                player.fields["main"][:, self.round_index, 0]
                > player.fields["main"][:, self.round_index - 1, 0]
            )[0]
            player.fields[card_type][batches_played, self.round_index - 1, :] = selected_cards[
                batches_played
            ]
        else:
            player.play_main_card(selected_cards, self.round_index)
            batches_played = torch.arange(batch_size, device=self.device)

        if self.replace_remaining_cards:
            selected_card_idx = torch.gather(indices, 1, index).squeeze(1)
            self.deck_availability[card_type][batches_played, selected_card_idx[batches_played]] = (
                False
            )
        else:
            self.deck_availability[card_type].scatter_(1, indices, False)

        if card_type == "bonus" and self.replace_remaining_cards and batches_played.shape[0] > 0:
            B = batches_played
            selected_pos = index[B].squeeze(1)
            discard_mask = torch.arange(self.draft_size, device=self.device).unsqueeze(
                0
            ) != selected_pos.unsqueeze(1)
            row_idx = B.unsqueeze(1).expand(-1, self.draft_size)[discard_mask]
            col_idx = indices[B][discard_mask]
            self.bonus_discard[row_idx, col_idx] = True

        probability = torch.ones(batch_size, device=self.device)
        probability[batches_played] = torch.gather(probabilities, 1, index).squeeze(1)[
            batches_played
        ]
        return probability

    def _solo_play_bonus(self, player: PlayerLike, temperature: float) -> torch.Tensor:
        batch_size = player.get_current_batch_size()
        return self._solo_play_card(player, "bonus", batch_size, temperature)

    def _step_round_multiplayer(self, players: Sequence[PlayerLike]) -> RoundResult:
        batch_size = players[0].get_current_batch_size()
        n_players = len(players)
        index_played_from_hand = torch.zeros(
            batch_size, n_players, dtype=torch.long, device=self.device
        )
        n_bonus_cards_to_draw = torch.zeros(
            batch_size, n_players, dtype=torch.long, device=self.device
        )

        for p, player in enumerate(players):
            _, index, selected_cards = player.evaluate_cards(
                player.cards_hand, self.round_index, mode="play"
            )
            player.play_main_card(selected_cards, self.round_index)
            index_played_from_hand[:, p] = index.squeeze()
            n_bonus_cards_to_draw[:, p] = (
                player.fields["main"][:, :, MAP_INDEX_IN_FLATTENED_CARD].sum(dim=1).long()
                + player.fields["bonus"][:, :, MAP_INDEX_IN_FLATTENED_CARD].sum(dim=1).long()
                + 1
            )

        last_card_played_ids = torch.stack(
            [p.fields["main"][:, self.round_index, 0] for p in players], dim=1
        )
        # Draft priority: who played lowest card ID (for RealNNGame stats)
        draft_winner_this_round = None
        if self.round_index < self.n_rounds - 1:
            draft_winner_this_round = last_card_played_ids.argmin(dim=1)  # (batch_size,)
        draft_pool = self._draw_draft_pool(batch_size, n_players)
        self._resolve_draft_batched(
            players, last_card_played_ids, draft_pool, index_played_from_hand
        )
        self._resolve_bonus_cards_batched(players, n_bonus_cards_to_draw)
        self.round_index += 1
        return RoundResult(picked_probabilities=[], draft_winner_this_round=draft_winner_this_round)

    def _draw_draft_pool(self, batch_size: int, n_players: int) -> torch.Tensor:
        card_length = self.decks["main"].shape[1]
        expanded = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)
        indices = torch.multinomial(
            self.deck_availability["main"].float(), n_players + 1, replacement=False
        )
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
        draft_pool = torch.gather(expanded, 1, indices_expanded)
        self.deck_availability["main"].scatter_(1, indices, False)
        return draft_pool

    def _resolve_draft_batched(
        self,
        players: Sequence[PlayerLike],
        last_card_played_ids: torch.Tensor,
        draft_pool: torch.Tensor,
        index_played_from_hand: torch.Tensor,
    ) -> None:
        if self.round_index >= self.n_rounds - 1:
            return
        batch_size = draft_pool.shape[0]
        n_draft_cards = draft_pool.shape[1]
        n_players = len(players)
        card_length = draft_pool.shape[2]

        all_logits_list = []
        for player in players:
            logits, _, _ = player.evaluate_cards(
                draft_pool, self.round_index, mode="draft", return_logits=True
            )
            all_logits_list.append(logits)
        all_logits = torch.stack(all_logits_list, dim=0)
        card_taken = torch.zeros(batch_size, n_draft_cards, dtype=torch.bool, device=self.device)
        priority_order = last_card_played_ids.argsort(dim=1)

        for rank in range(n_players):
            player_indices = priority_order[:, rank]
            batch_indices = torch.arange(batch_size, device=self.device)
            current_logits = all_logits[player_indices, batch_indices, :]
            current_logits = current_logits.masked_fill(card_taken, float("-inf"))
            selected_indices = current_logits.argmax(dim=1)
            card_taken.scatter_(1, selected_indices.unsqueeze(1), True)
            selected_indices_expanded = (
                selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, card_length)
            )
            selected_cards = torch.gather(draft_pool, 1, selected_indices_expanded).squeeze(1)
            for p in range(n_players):
                mask = player_indices == p
                if mask.any():
                    batch_p = mask.nonzero(as_tuple=True)[0]
                    hand_p = index_played_from_hand[batch_p, p]
                    players[p].cards_hand[batch_p, hand_p, :] = selected_cards[batch_p]

    def _resolve_bonus_cards_batched(
        self,
        players: Sequence[PlayerLike],
        n_bonus_cards_to_draw: torch.Tensor,
    ) -> None:
        if self.round_index == 0:
            return
        card_length = self.decks["bonus"].shape[1]

        for player_idx, player in enumerate(players):
            triggers_bonus = (
                player.fields["main"][:, self.round_index, 0]
                > player.fields["main"][:, self.round_index - 1, 0]
            )
            if not triggers_bonus.any():
                continue
            game_indices = triggers_bonus.nonzero().squeeze(1)
            n_games = len(game_indices)
            n_cards_per_game = n_bonus_cards_to_draw[game_indices, player_idx]
            max_cards = int(n_cards_per_game.max().item())
            if max_cards == 0:
                continue

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
                self.reshuffle_bonus_discard_if_needed(
                    game_ids=torch.tensor([gid], device=self.device),
                    n_cards_needed=n_to_draw,
                )
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
                ).squeeze(0)
                self.deck_availability["bonus"][gid].scatter_(0, indices, False)
                drawn_cards = self.decks["bonus"][indices]
                all_bonus_cards[i, :n_to_draw, :] = drawn_cards
                valid_mask[i, :n_to_draw] = True
                drawn_indices_padded[i, :n_to_draw] = indices

            logits, _, _ = player.evaluate_cards(
                all_bonus_cards,
                self.round_index,
                mode="bonus",
                games_indices=game_indices,
                return_logits=True,
            )
            logits = logits.masked_fill(~valid_mask, float("-inf"))
            selected_indices = logits.argmax(dim=1)
            selected_indices_expanded = (
                selected_indices.unsqueeze(1).unsqueeze(2).expand(-1, 1, card_length)
            )
            selected_cards = torch.gather(all_bonus_cards, 1, selected_indices_expanded).squeeze(1)
            player.fields["bonus"][game_indices, self.round_index - 1, :] = selected_cards
            discarded_mask = valid_mask & (
                torch.arange(max_cards, device=self.device).unsqueeze(0)
                != selected_indices.unsqueeze(1)
            )
            if discarded_mask.any():
                row_idx = game_indices.unsqueeze(1).expand(-1, max_cards)[discarded_mask]
                col_idx = drawn_indices_padded[discarded_mask]
                self.bonus_discard[row_idx, col_idx] = True

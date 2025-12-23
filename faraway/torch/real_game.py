"""
Real batched games using nn bots.
"""

import sys
from typing import Annotated

import torch
import typer
from loguru import logger

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import MainCard, MainCardsSeries
from faraway.core.human_player import HumanPlayer
from faraway.torch.base_game import BaseNNGame
from faraway.torch.nn_player import NNPlayer

MAP_INDEX_IN_FLATTENED_CARD = MainCard.get_field_index("map", "assets")


class RealNNGame(BaseNNGame):
    def __init__(
        self,
        players: list[BasePlayer],
        n_rounds: int = 8,
        use_bonus_cards: bool = True,
        device: torch.device | None = None,
        verbose: int = 0,
    ):
        super().__init__(n_rounds, use_bonus_cards, device, players, verbose)
        self.verbose = verbose
        self.draft_pool: torch.Tensor | None = None

    def reset_games_batch(self, batch_size: int) -> None:
        # reset the deck and the fields
        super().reset_games_batch(batch_size)
        # expand the main deck to batch size: (68, 24) -> (batch_size, 68, 24)
        expanded_main_deck = self.decks["main"].unsqueeze(0).expand(batch_size, -1, -1)
        card_length = self.decks["main"].shape[1]
        # give 3 main cards to each player
        for player in self.players:
            # draw 3 main cards - indices shape: (batch_size, 3) with values 0-67
            indices = torch.multinomial(
                self.deck_availability["main"].float(), 3, replacement=False
            )
            # expand indices for gather: (batch_size, 3) -> (batch_size, 3, 24)
            indices_expanded = indices.unsqueeze(2).expand(-1, -1, card_length)
            # gather cards: (batch_size, 68, 24) -> (batch_size, 3, 24)
            player.cards_hand = torch.gather(expanded_main_deck, 1, indices_expanded)
            # update the deck availability
            self.deck_availability["main"].scatter_(1, indices, False)
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
        draft_pool = self.draw_draft_pool()
        if self.verbose > 98:
            logger.debug("RESOLVE ACTIONS ONE GAME")
        for i in range(batch_size):
            if self.verbose > 98:
                logger.debug(f"Game #{i}")
            self.resolve_actions_one_game(
                last_card_played_ids[i, :],
                draft_pool[i : i + 1, :, :],
                index_played_from_hand[i, :],
                n_bonus_cards_to_draw[i, :],
                game_id=i,
            )
        self.round_index += 1

    def resolve_actions_one_game(
        self,
        last_card_played_ids: torch.Tensor,  # (n_players,)
        draft_pool: torch.Tensor,  # (n_players + 1, card_length)
        index_played_from_hand: torch.Tensor,  # (n_players,)
        n_bonus_cards_to_draw: torch.Tensor,  # (n_players,)
        game_id: int,
    ) -> None:
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
        player: NNPlayer,
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
                > self.players[0].fields["main"][game_id, self.round_index - 1, 0]
            ):
                # cap to available bonus cards
                n_available = self.deck_availability["bonus"][game_id, :].sum().item()
                if n_bonus_cards_to_draw > n_available:
                    raise ValueError(
                        f"Player {player} can't draw {n_bonus_cards_to_draw} "
                        f"bonus cards, only {n_available} available"
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

    def run_tournament(self, n_batches: int, batch_size: int) -> tuple[list[int], list[float]]:
        scores = self.play_games_batches(  # (n_batches * batch_size, n_players)
            n_batches,
            batch_size,
        )
        winner = scores.argmax(dim=1)
        # count the number of occurence of each player id in the winner tensor
        wins = []
        win_rate = []
        for player_id in range(len(self.players)):
            wins.append(torch.where(winner == player_id)[0].shape[0])
            win_rate.append(wins[-1] / (n_batches * batch_size) * 100)
        # average scores for each player
        mean_scores = scores.mean(dim=0).tolist()
        if self.verbose > 0:
            logger.info(
                f"Tournament completed.\n"
                f"Mean scores: {mean_scores.tolist()}\n"
                f"Wins: {wins}\n"
                f"Win rate: {win_rate}%\n"
            )
        return wins, mean_scores


def main(
    players: Annotated[
        list[str],
        typer.Argument(help="Paths to the players to use (e.g., model.pt or 'human' or 'random')"),
    ],
    log_to_file: Annotated[bool, typer.Option(help="Whether to log to a file")] = False,
    batch_size: Annotated[int, typer.Option(help="Training batch size")] = 32,
    n_batches: Annotated[int, typer.Option(help="Number of batches to play")] = 100,
    verbose: Annotated[int, typer.Option(help="Verbosity level")] = 1,
) -> None:
    """Run a tournament between NN players."""
    logger.remove()  # remove default stderr handler
    if log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)
    n_rounds = 8

    # load the players
    players_list: list[BasePlayer] = []
    for player in players or []:
        if player.endswith(".pt"):
            players_list.append(NNPlayer.load(player))
        elif player == "human":
            players_list.append(HumanPlayer(n_rounds))
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
    )
    game.run_tournament(n_batches=n_batches, batch_size=batch_size)


if __name__ == "__main__":
    typer.run(main)

"""
Real batched games using nn bots.
"""

import sys
from collections.abc import Sequence
from typing import Annotated, cast

import torch
import torch as torch_module
import typer
from loguru import logger

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import MainCard, MainCardsSeries
from faraway.core.human_player import HumanPlayer
from faraway.torch.base_game import BaseNNGame
from faraway.torch.env import BatchedFarawayEnv, PlayerLike
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
        self._env = BatchedFarawayEnv(
            n_rounds=self.n_rounds,
            use_bonus_cards=self.use_bonus_cards,
            draft_size=len(players) + 1,
            use_hand=True,
            n_cards_hand=3,
            replace_remaining_cards=True,
            device=self.device,
            verbose=self.verbose,
        )

    def reset_games_batch(self, batch_size: int) -> None:
        self._env.reset(batch_size, cast(Sequence[PlayerLike], self.players))
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index
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

    def play_round(self) -> None:
        result = self._env.step_round(cast(Sequence[PlayerLike], self.players))
        self.deck_availability = self._env.deck_availability
        self.bonus_discard = self._env.bonus_discard
        self.round_index = self._env.round_index
        if result.draft_winner_this_round is not None:
            batch_size = result.draft_winner_this_round.shape[0]
            self.draft_priority_wins.scatter_add_(
                1,
                result.draft_winner_this_round.unsqueeze(1),
                torch.ones(batch_size, 1, dtype=torch.long, device=self.device),
            )

    def get_scores(self) -> torch.Tensor:
        return self._env.get_scores(cast(Sequence[PlayerLike], self.players))

    def get_bonus_cards_played(self) -> torch.Tensor:
        return self._env.get_bonus_cards_played(cast(Sequence[PlayerLike], self.players))

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

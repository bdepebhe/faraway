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
    Batched solo play game using tensors.

    Attributes:
        n_rounds: Number of rounds in the game
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
        use_draft: bool = False,
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
        self.use_draft = use_draft
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
        self.step_id = 0  # step id for TensorBoard

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

        if self.use_draft:
            # Deal initial hand to the solo player
            self.deal_initial_hands(n_cards=self.n_cards_hand, batch_size=batch_size)

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

    def log_hparams(self, extra_hparams: dict[str, Any] | None = None) -> None:
        """Log hyperparameters to TensorBoard for experiment comparison."""
        hparams = {
            "player_type": self.player_type,
            "n_rounds": self.n_rounds,
            "draft_size": self.draft_size,
            "use_draft": self.use_draft,
            "n_cards_hand": self.n_cards_hand,
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
            # Use add_text instead of add_hparams to avoid creating timestamp subdirectories
            hparams_text = "\n".join(f"**{k}**: {v}" for k, v in hparams.items())
            self.writer.add_text("hparams", hparams_text, 0)

    def _play_from_hand(self) -> torch.Tensor:
        """Choose and play a card from hand. Returns probability of the selection.

        Used when use_draft=True. The model selects one card from the hand to play.
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

        Used when use_draft=True. The model selects one card from the river to add to hand.
        """
        player = self.players[0]
        batch_size = player.get_current_batch_size()

        # Check if there are enough cards in the deck
        available_counts = self.deck_availability["main"].sum(dim=1)  # (batch,)
        can_draft = available_counts >= self.draft_size

        if not can_draft.any():
            # No drafting possible (late game), return prob=1 (no decision)
            return torch.ones(batch_size, device=self.device)

        # Sample cards from the deck for the river
        # For batches that can't draft, we still sample but won't use the result
        # Use min of draft_size and available cards to avoid errors
        indices = torch.multinomial(
            self.deck_availability["main"].float(),
            min(self.draft_size, int(available_counts.min().item())),
            replacement=False,
        )  # (batch, draft_size or less)

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

        # Get probability (1.0 for batches that couldn't draft)
        probability = torch.ones(batch_size, device=self.device)
        probability[can_draft] = torch.gather(probabilities, 1, index).squeeze(1)[can_draft]
        return probability

    def _play_card(self, type: str) -> torch.Tensor:
        """Play a card of the given type (main or bonus).

        For main cards with use_draft=False: samples from deck and plays directly.
        For main cards with use_draft=True: handled by _play_from_hand/_draft_to_hand.
        For bonus cards: always samples from bonus deck.
        """
        batch_size = self.players[0].get_current_batch_size()

        # For bonus cards, check if we need to reshuffle discarded cards
        # Per official rules: "If the Sanctuary deck is empty, shuffle the
        # discarded Sanctuary cards to form a new deck."
        if type == "bonus":
            self.reshuffle_bonus_discard_if_needed(n_cards_needed=self.draft_size)

        # select indices of cards to sample from the main deck
        indices = torch.multinomial(
            self.deck_availability[type].float(), self.draft_size, replacement=False
        )  # (batch, draft_size)
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

        if self.use_draft:
            # Draft mode: play from hand, then draft to refill hand
            picked_probability_play = self._play_from_hand()
            probabilities_list.append(picked_probability_play)

            # Draft a new card to hand (if cards remain in deck)
            picked_probability_draft = self._draft_to_hand()
            probabilities_list.append(picked_probability_draft)
        else:
            # No draft: pick directly from deck (original behavior)
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
        # empty probas tensor for training. dim 0 is batch, but no values
        self.picked_probabilities = torch.ones(
            self.rl_params["train_batch_size"], 0, device=self.device
        )
        # play the game and get the log probabilities
        self.play_games_batch(self.rl_params["train_batch_size"], learning_mode=True)
        log_probs = torch.log(self.picked_probabilities)  # (batch, n_rounds)
        # apply final_count_from_tensor_field for each element of the batch
        scores = self.get_scores()[:, 0]
        advantage = scores - self.baseline

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
            self.writer.add_scalar("baseline/value", self.baseline, n_training_games_played)
            self.writer.add_scalar(
                "advantage/mean", advantage.mean().item(), n_training_games_played
            )
            self.writer.add_scalar("advantage/std", advantage.std().item(), n_training_games_played)
            self.writer.add_scalar("loss/policy", loss.item(), n_training_games_played)
            self.writer.add_scalar("step_id", self.step_id, n_training_games_played)

        if self.verbose > 0:
            logger.info(
                f"Step {self.step_id}. "
                f"Score: {scores.mean().item():.2f}. "
                f"Baseline: {self.baseline:.2f}. "
                f"Loss: {loss.item():.2f}. "
                f"Games: {n_training_games_played}"
            )
        # update the baseline
        self.baseline = (
            self.baseline + self.rl_params["update_baseline_rate"] * (scores.mean() - self.baseline)
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
    use_draft: Annotated[
        bool, typer.Option(help="Enable draft mechanism (hand management)")
    ] = False,
    n_cards_hand: Annotated[
        int, typer.Option(help="Number of cards in hand (when using draft)")
    ] = 3,
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
            "use_cards_hand_in_state": False,
            "use_mode_embedding": use_mode_embedding,
        }
    else:
        raise ValueError(f"Unknown player type: {player_type}")

    rl_params: dict[str, Any] = {
        "prior_baseline_score": 29,
        "train_batch_size": batch_size,
        "update_baseline_rate": baseline_update_rate,
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
        use_draft=use_draft,
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

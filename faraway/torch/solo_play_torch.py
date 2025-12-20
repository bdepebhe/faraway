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

import argparse
import sys
from datetime import datetime
from typing import Any

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from faraway.torch.base_game import BaseNNGame
from faraway.torch.models import create_mlp_model
from faraway.torch.player import NNPlayer


def sample_cards_from_availability_tensor(
    availability_tensor: torch.Tensor, draft_size: int
) -> torch.Tensor:
    # the availability tensor is a boolean tensor of shape (batch, n_cards)
    # we want to return a tensor of size (batch, draft_size) with the indices of the sampled cards
    # we can use torch.multinomial to sample the indices with replacement
    return torch.multinomial(availability_tensor, draft_size, replacement=False)


class SoloNNGame(BaseNNGame):
    """
    Batched solo play game using tensors.

    Attributes:
        batch_size: Number of parallel games
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
        model_path: str | None = None,
        verbose: int = 1,
        prior_baseline_score: float = 29,
        update_baseline_rate: float = 0.05,
        batch_size: int = 32,
        device: torch.device | None = None,
        hidden_layers_sizes: list[int] = [512, 1024, 512],  # noqa: B006
        dropout_rate: float = 0.1,
        experiment_name: str | None = None,
        log_dir: str = "runs",
    ):
        super().__init__(n_rounds, use_bonus_cards, batch_size, device)
        self.draft_size = draft_size
        self.replace_remaining_cards = replace_remaining_cards
        self.verbose = verbose
        self.prior_baseline_score = prior_baseline_score
        self.update_baseline_rate = update_baseline_rate
        self.hidden_layers_sizes = hidden_layers_sizes
        self.dropout_rate = dropout_rate

        # TensorBoard monitoring
        # total_games_played is the "epoch" metric - counts environment interactions
        # This allows fair comparison between experiments with different batch sizes
        if experiment_name is None:
            experiment_name = f"faraway_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        self.writer = SummaryWriter(log_dir=f"{log_dir}/{experiment_name}")
        self.reset_learning()

    def reset_learning(self, model_path: str | None = None) -> None:
        self.total_games_played = 0
        self.baseline = self.prior_baseline_score
        self.step_id = 0  # step id for TensorBoard
        self.model = create_mlp_model(
            self.nn_input_size, model_path, self.hidden_layers_sizes, self.dropout_rate
        )
        self.players = [
            NNPlayer(
                self.model, self.device, self.batch_size, self.n_main_cards, self.n_bonus_cards
            )
        ]  # only one player for solo play
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def dump_model(self, model_path: str) -> None:
        torch.save(self.model, model_path)

    def close_tensorboard(self) -> None:
        """Close the TensorBoard writer. Call this when training is complete."""
        self.writer.close()

    def log_hparams(self, extra_hparams: dict[str, Any] | None = None) -> None:
        """Log hyperparameters to TensorBoard for experiment comparison."""
        hparams = {
            "batch_size": self.batch_size,
            "n_rounds": self.n_rounds,
            "draft_size": self.draft_size,
            "prior_baseline_score": self.prior_baseline_score,
            "use_bonus_cards": self.use_bonus_cards,
            "replace_remaining_cards": self.replace_remaining_cards,
            # "hidden_layers_sizes": self.hidden_layers_sizes,
            "dropout_rate": self.dropout_rate,
        }
        if extra_hparams:
            hparams.update(extra_hparams)
        self.writer.add_hparams(hparams, {})

    def _play_card(self, type: str) -> torch.Tensor:
        # select indices of cards to sample from the main deck
        indices = torch.multinomial(
            self.deck_availability[type].float(), self.draft_size, replacement=False
        )  # (batch, draft_size)
        # sample the cards
        possible_cards_tensor = self.decks[type][indices]  # (batch, draft_size, MainCard.length())
        # prepare the state tensor
        fields = torch.concat(
            [self.players[0].fields["main"], self.players[0].fields["bonus"]], dim=1
        )
        flattened_state = torch.flatten(fields, start_dim=1)  # (batch, (8+6)*24)
        # append the round index
        round_index_tensor = torch.tensor(
            [[self.round_index] * self.batch_size], device=self.device
        ).T
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
        if type == "bonus":
            # for playing a bonus card, it depends on the previous main card
            batches_indices_where_card_played = torch.where(
                self.players[0].fields["main"][:, self.round_index, 0]
                > self.players[0].fields["main"][:, self.round_index - 1, 0]
            )[0]
        elif type == "main":
            batches_indices_where_card_played = torch.arange(self.batch_size, device=self.device)
        position_to_insert = {"main": self.round_index, "bonus": self.round_index - 1}[type]
        # add the card to the field except for last bonus card
        self.players[0].fields[type][batches_indices_where_card_played, position_to_insert, :] = (
            selected_cards[batches_indices_where_card_played]
        )
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
        # get the probability of the selected card
        # (1 if no card played, so log(1)=0 won't affect the loss)
        probability = torch.ones(self.batch_size, device=self.device)
        probability[batches_indices_where_card_played] = torch.gather(
            probabilities, 1, index
        ).squeeze(1)[batches_indices_where_card_played]  # (batch,)
        return probability

    def play_round(self) -> torch.Tensor:
        # play the main card and
        picked_probability_main = self._play_card("main")
        # play the bonus card
        if self.use_bonus_cards and self.round_index > 0:
            picked_probability_bonus = self._play_card("bonus")
            # concatenate the probabilities
            picked_probabilities = torch.stack(
                [picked_probability_main, picked_probability_bonus], dim=1
            )  # (batch, 2)
        else:
            picked_probabilities = picked_probability_main.unsqueeze(1)
        # increment the round index
        self.round_index += 1
        return picked_probabilities

    def learning_step(self) -> None:
        # empty probas tensor for training. dim 0 is batch, but no values
        self.picked_probabilities = torch.ones(self.batch_size, 0, device=self.device)
        # play the game and get the log probabilities
        self.play_games_batch(learning_mode=True)
        log_probs = torch.log(self.picked_probabilities)  # (batch, n_rounds)
        # apply final_count_from_tensor_field for each element of the batch
        scores = self.get_scores()[:, 0]
        advantage = scores - self.baseline

        loss = (-torch.sum(log_probs, 1) * advantage).mean()  # scalar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update total games played (epoch metric based on environment interactions)
        self.total_games_played += self.batch_size

        # Log metrics to TensorBoard
        # Using total_games_played as x-axis for fair comparison across batch sizes
        self.writer.add_scalar("score/mean", scores.mean().item(), self.total_games_played)
        self.writer.add_scalar("score/max", scores.max().item(), self.total_games_played)
        self.writer.add_scalar("score/min", scores.min().item(), self.total_games_played)
        self.writer.add_scalar("score/std", scores.std().item(), self.total_games_played)
        self.writer.add_scalar("baseline/value", self.baseline, self.total_games_played)
        self.writer.add_scalar("advantage/mean", advantage.mean().item(), self.total_games_played)
        self.writer.add_scalar("advantage/std", advantage.std().item(), self.total_games_played)
        self.writer.add_scalar("loss/policy", loss.item(), self.total_games_played)
        self.writer.add_scalar("step_id", self.step_id, self.total_games_played)

        if self.verbose > 0:
            logger.info(
                f"Step {self.step_id}. "
                f"Score: {scores.mean().item():.2f}. "
                f"Baseline: {self.baseline:.2f}. "
                f"Loss: {loss.item():.2f}. "
                f"Games: {self.total_games_played}"
            )
        # update the baseline
        self.baseline = (
            self.baseline + self.update_baseline_rate * (scores.mean() - self.baseline)
        ).item()
        self.step_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_to_file", action="store_true", help="Whether to log to a file")
    parser.add_argument(
        "--experiment_name", type=str, default=None, help="Name for TensorBoard experiment"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--n_steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument(
        "--n_eval_batches", type=int, default=100, help="Number of evaluation batches"
    )
    args = parser.parse_args()

    if args.log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    game = SoloNNGame(
        verbose=2,
        prior_baseline_score=29,
        batch_size=args.batch_size,
        experiment_name=args.experiment_name,
        dropout_rate=0.1,
        draft_size=10,
        hidden_layers_sizes=[512, 1024, 512],
    )

    # Log hyperparameters for experiment comparison
    game.log_hparams({"n_steps": args.n_steps, "learning_rate": 0.0005})

    # Initial evaluation
    initial_scores = game.play_games_batches(n_batches=args.n_eval_batches)
    logger.info(
        f"Initial score: {initial_scores.mean().item():.2f}. Best: {initial_scores.max().item()}"
    )

    # Training
    game.batch_size = args.batch_size
    for _ in range(args.n_steps):
        game.learning_step()

    # Final evaluation
    final_scores = game.play_games_batches(n_batches=args.n_eval_batches)
    logger.info(f"Final score: {final_scores.mean().item():.2f}. Best: {final_scores.max().item()}")

    # Log final metrics
    game.writer.add_scalar("eval/initial_score", initial_scores.mean().item(), 0)
    game.writer.add_scalar("eval/final_score", final_scores.mean().item(), game.total_games_played)

    game.dump_model(f"runs/{game.experiment_name}/faraway_model.pt")
    game.close_tensorboard()
    print(f"\nTensorBoard logs saved to: runs/{game.experiment_name}")
    print("Run 'tensorboard --logdir=runs' to view results")

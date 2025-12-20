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

import torch
import torch.nn as nn
from loguru import logger

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


def final_count_from_tensor_field(
    main_field_tensor: torch.Tensor, bonus_field_tensor: torch.Tensor
) -> torch.Tensor:
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


def batched_final_count(main_fields: torch.Tensor, bonus_fields: torch.Tensor) -> torch.Tensor:
    n_rounds = main_fields.shape[1]
    # concatenate the main and bonus fields
    fields = torch.concat([main_fields, bonus_fields], dim=1)
    # map the function along axis 0
    return torch.stack(
        [final_count_from_tensor_field(fields[i, :], n_rounds) for i in range(fields.shape[0])]
    )


def sample_cards_from_availability_tensor(
    availability_tensor: torch.Tensor, draft_size: int
) -> torch.Tensor:
    # the availability tensor is a boolean tensor of shape (batch, n_cards)
    # we want to return a tensor of size (batch, draft_size) with the indices of the sampled cards
    # we can use torch.multinomial to sample the indices with replacement
    return torch.multinomial(availability_tensor, draft_size, replacement=False)


class SoloNNGame:
    """
    Batched game using tensors.

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
        batch_size: int = 32,
        device: torch.device | None = None,
        hidden_layers_sizes: list[int] = [512, 512],  # noqa: B006
        dropout_rate: float = 0.1,
    ):
        self.n_rounds = n_rounds
        self.draft_size = draft_size
        self.replace_remaining_cards = replace_remaining_cards
        self.use_bonus_cards = use_bonus_cards
        self.verbose = verbose
        self.prior_baseline_score = prior_baseline_score
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_deck = get_main_deck_tensor()
        bonus_deck = get_bonus_deck_tensor()
        self.decks = {
            "main": main_deck,
            "bonus": bonus_deck,
        }
        self.n_main_cards = n_rounds
        if self.use_bonus_cards:
            self.n_bonus_cards = n_rounds - 1
        else:
            self.n_bonus_cards = 0
        self.state_length = (
            MainCard.length() * (self.n_main_cards + self.n_bonus_cards)  # previous cards
            + 1  # round index
        )
        self.nn_input_size = self.state_length + MainCard.length()  # current card to choose
        self.model = self._create_model(model_path, hidden_layers_sizes, dropout_rate)

        self.reset_games_batch()
        self.baseline = self.prior_baseline_score  # Initialize once, persists across learning steps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

    def _create_model(
        self,
        model_path: str | None = None,
        hidden_layers_sizes: list[int] = [512, 512],  # noqa: B006
        dropout_rate: float = 0.1,
    ) -> nn.Module:
        if model_path is not None:
            model = torch.load(model_path)
            if model.input_size != self.nn_input_size:
                raise ValueError(
                    f"Input size of model {model_path} {model.input_size} "
                    f"does not match expected input size {self.nn_input_size}"
                )
            return model
        layers_sizes = [self.nn_input_size] + hidden_layers_sizes
        layers = []
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layers_sizes[-1], 1))
        return nn.Sequential(*layers)

    def reset_games_batch(self) -> None:
        main_deck_availability = torch.ones(
            self.batch_size, self.decks["main"].shape[0], dtype=torch.bool, device=self.device
        )  # (batch, n_main_cards)
        bonus_deck_availability = torch.ones(
            self.batch_size, self.decks["bonus"].shape[0], dtype=torch.bool, device=self.device
        )  # (batch, n_bonus_cards)
        self.deck_availability = {
            "main": main_deck_availability,
            "bonus": bonus_deck_availability,
        }
        main_fields = torch.zeros(
            self.batch_size, self.n_main_cards, MainCard.length(), device=self.device
        )  # (batch, n_main_cards, MainCard.length())
        bonus_fields = torch.zeros(
            self.batch_size, self.n_bonus_cards, MainCard.length(), device=self.device
        )  # (batch, n_bonus_cards, MainCard.length())
        self.fields = {
            "main": main_fields,
            "bonus": bonus_fields,
        }
        self.round_index = 0
        # empty probas tensor for training. dim 0 is batch, but no values
        self.picked_probabilities = torch.ones(self.batch_size, 0, device=self.device)

    def dump_model(self, model_path: str) -> None:
        torch.save(self.model, model_path)

    def _play_card(self, type: str) -> torch.Tensor:
        # select indices of cards to sample from the main deck
        indices = torch.multinomial(
            self.deck_availability[type].float(), self.draft_size, replacement=False
        )  # (batch, draft_size)
        # sample the cards
        possible_cards_tensor = self.decks[type][indices]  # (batch, draft_size, MainCard.length())
        # prepare the state tensor
        fields = torch.concat([self.fields["main"], self.fields["bonus"]], dim=1)
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
                self.fields["main"][:, self.round_index, 0]
                > self.fields["main"][:, self.round_index - 1, 0]
            )[0]
        elif type == "main":
            batches_indices_where_card_played = torch.arange(self.batch_size, device=self.device)
        position_to_insert = {"main": self.round_index, "bonus": self.round_index - 1}[type]
        # add the card to the field except for last bonus card
        self.fields[type][batches_indices_where_card_played, position_to_insert, :] = (
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
        # add the probabilities to the picked_probabilities tensor
        self.picked_probabilities = torch.concat(
            [self.picked_probabilities, picked_probabilities], dim=1
        )
        # increment the round index
        self.round_index += 1
        return picked_probabilities

    def play_game(self) -> torch.Tensor:
        self.reset_games_batch()
        for _ in range(self.n_rounds):
            self.play_round()
        return self.picked_probabilities

    def get_score(self) -> torch.Tensor:
        return torch.tensor(
            [
                final_count_from_tensor_field(
                    self.fields["main"][i, :, :], self.fields["bonus"][i, :, :]
                )
                for i in range(self.batch_size)
            ],
            dtype=torch.float32,
            device=self.device,
        )  # (batch,)

    def learning_step(self) -> None:
        log_probs = torch.log(self.play_game())  # (batch, n_rounds)
        # apply final_count_from_tensor_field for each element of the batch
        scores = self.get_score()
        advantage = scores - torch.tensor(self.baseline, device=self.device)

        # update the baseline
        self.baseline = self.baseline + 0.05 * (scores.mean() - self.baseline)

        loss = (-torch.sum(log_probs, 1) * advantage).mean()  # scalar
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.verbose > 0:
            logger.info(
                f"Score: {scores.mean().item()}. Baseline: "
                f"{self.baseline:.2f}. Loss: {loss.item():.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_to_file", action="store_true", help="Whether to log to a file")
    args = parser.parse_args()

    if args.log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    game = SoloNNGame(verbose=2, prior_baseline_score=29)
    game.batch_size = 1000
    initial_play = game.play_game()
    initial_score = game.get_score()
    print(f"Initial score: {initial_score.mean().item()}. Best score: {initial_score.max().item()}")
    game.batch_size = 32
    for _ in range(1000):
        game.learning_step()
    game.batch_size = 1000
    final_play = game.play_game()
    final_score = game.get_score()
    print(f"Final score: {final_score.mean().item()}. Best score: {final_score.max().item()}")
    game.dump_model("faraway_model.pt")

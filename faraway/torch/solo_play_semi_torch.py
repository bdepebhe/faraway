import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from faraway.core.data_structures import MainCard
from faraway.core.solo_play import SoloPlay


class TorchSimpleSoloPlay(SoloPlay):
    def __init__(
        self,
        n_rounds: int = 8,
        use_bonus_cards: bool = True,
        model_path: str | None = None,
        verbose: int = 1,
        prior_baseline_score: float = 13,
        device: torch.device | None = None,
    ):
        self.state_length = (
            MainCard.length() * (n_rounds - 1)  # previous card
            + 1  # index of the current round
        )
        super().__init__(n_rounds, use_bonus_cards, verbose)
        self.model = self._create_model(model_path)
        self.prior_baseline_score = prior_baseline_score
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.00001)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reset(self) -> None:
        super().reset()
        self.state = self._initialize_state()
        self.log_probs: list[torch.Tensor] = []

    def _create_model(self, model_path: str | None = None) -> nn.Module:
        input_size = (
            self.state_length + MainCard.length()  # current card to choose
        )
        if model_path is not None:
            model = torch.load(model_path)
            if model.input_size != input_size:
                raise ValueError(
                    f"Input size of model {model_path} {model.input_size} does "
                    f"not match expected input size {input_size}"
                )
            return model
        return nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def dump_model(self, model_path: str) -> None:
        torch.save(self.model, model_path)

    def _initialize_state(self) -> torch.Tensor:
        return torch.zeros(self.state_length)

    def _play_round(self) -> None:
        round_index = len(self.player_field.main_cards)
        # tensor of possible cards for the current round
        possible_cards = torch.stack(
            [torch.tensor(card.flatten()) for card in self.main_deck]
        )  # (N_possible_card, MainCard.length())
        state_expanded = self.state.unsqueeze(0).expand(
            possible_cards.shape[0], -1
        )  # (N_possible_card, state_length)
        input_tensor = torch.concat(
            [state_expanded, possible_cards], dim=1
        )  # (N_possible_card, state_length + MainCard.length())
        logits = self.model(input_tensor).squeeze()  # (N_possible_card,)
        # take softmax to get probabilities
        probabilities = torch.softmax(logits, dim=0)  # (N_possible_card,)
        # sample from probabilities
        index = torch.multinomial(probabilities, 1).item()
        main_card = self.main_deck.pop(index)
        self.player_field.main_cards.append(main_card)
        # update state
        self.log_probs.append(torch.log(probabilities[index]))
        if round_index < self.n_rounds - 1:
            self.state[-1] = round_index + 1
            self.state[round_index * MainCard.length() : (round_index + 1) * MainCard.length()] = (
                torch.tensor(main_card.flatten())
            )

    def learning_step(self) -> None:
        score = self.play()
        advantage = score - self.prior_baseline_score
        loss = -torch.sum(torch.stack(self.log_probs)) * advantage
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.verbose > 0:
            logger.info(f"Learning step completed. Score: {score}. Loss: {loss.item()}")


if __name__ == "__main__":
    # CLI for n_simulations and type of player
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_simulations", type=int, default=1_000, help="Number of simulations to run"
    )
    parser.add_argument("--player_type", type=str, default="random", help="Type of player to use")
    parser.add_argument(
        "--no_bonus_cards", action="store_true", help="Number of bonus cards to use"
    )
    parser.add_argument("--log_to_file", action="store_true", help="Whether to log to a file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the model to load")
    parser.add_argument(
        "--prior_baseline_score", type=float, default=13, help="Prior baseline score"
    )
    parser.add_argument(
        "--n_learning_steps", type=int, default=0, help="Number of learning steps. Ignored if 0"
    )
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    if args.log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)

    solo_play = TorchSimpleSoloPlay(
        verbose=args.verbose, use_bonus_cards=not args.no_bonus_cards, model_path=args.model_path
    )
    for _ in range(args.n_learning_steps):
        solo_play.learning_step()
    results = solo_play.run_multiple_simulations(n_simulations=args.n_simulations)
    if args.verbose:
        logger.info(f"Average score: {np.mean(results)}")
        logger.info(f"Minimum score: {np.min(results)}")
        logger.info(f"Maximum score: {np.max(results)}")

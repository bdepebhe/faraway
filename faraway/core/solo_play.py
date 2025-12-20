import argparse
import sys
from abc import ABC, abstractmethod

import numpy as np
from loguru import logger

from faraway.core.final_count import final_count
from faraway.core.load_cards import load_bonus_cards, load_main_cards
from faraway.core.player_field import PlayerField


class SoloPlay(ABC):
    def __init__(self, n_rounds: int = 8, use_bonus_cards: bool = True, verbose: int = 1):
        self.n_rounds = n_rounds
        self.use_bonus_cards = use_bonus_cards
        self.verbose = verbose
        self.reset()

    def reset(self) -> None:
        self.player_field = PlayerField(n_rounds=self.n_rounds)
        self.main_deck = load_main_cards()
        if self.use_bonus_cards:
            self.bonus_deck = load_bonus_cards()
        else:
            self.bonus_deck = []

    def play(self) -> int:
        self.reset()
        for _ in range(self.n_rounds):
            self._play_round()
        if not self.player_field.validate_final_field(use_bonus_cards=self.use_bonus_cards):
            raise ValueError("Final field is not valid")
        return final_count(self.player_field)

    @abstractmethod
    def _play_round(self) -> None:
        # draw and play main card
        pass

    def run_multiple_simulations(self, n_simulations: int = 10) -> list[int]:
        results = []
        best_score = 0
        if self.verbose > 0:
            logger.info(f"Running {n_simulations} simulations")
            logger.info(f"Player type: {self.__class__.__name__}")
            logger.info(f"Using bonus cards: {self.use_bonus_cards}")
            logger.info(f"N rounds: {self.n_rounds}")
        for _ in range(n_simulations):
            results.append(self.play())
            if self.verbose > 1:
                logger.info(f"Simulation {_ + 1} completed")
            if results[-1] > best_score:
                best_score = results[-1]
                best_card_set = self.player_field
        if self.verbose:
            logger.info(f"Average score: {np.mean(results)}")
            logger.info(f"Minimum score: {np.min(results)}")
            logger.info(f"Maximum score: {np.max(results)}")
        if self.verbose > 1:
            logger.info(f"Best card set: {best_card_set}")
        return results


class RandomSoloPlay(SoloPlay):
    def _play_round(self) -> None:
        # draw and play main card
        index = np.random.randint(0, len(self.main_deck))
        main_card = self.main_deck.pop(index)
        self.player_field.main_cards.append(main_card)
        # draw and play bonus card if applicable
        if (
            self.use_bonus_cards
            and len(self.player_field.main_cards) >= 2
            and self.player_field.main_cards[-1].id > self.player_field.main_cards[-2].id
        ):
            index = np.random.randint(0, len(self.bonus_deck))
            bonus_card = self.bonus_deck.pop(index)
            self.player_field.bonus_cards.append(bonus_card)


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
    parser.add_argument("--verbose", type=int, default=1, help="Verbosity level")
    args = parser.parse_args()

    if args.log_to_file:
        logger.add("faraway.log")
    else:
        logger.add(sys.stdout)
    if args.player_type == "random":
        solo_play = RandomSoloPlay(verbose=args.verbose, use_bonus_cards=not args.no_bonus_cards)
    else:
        raise ValueError(f"Invalid player type: {args.player_type}")
    results = solo_play.run_multiple_simulations(n_simulations=args.n_simulations)
    if args.verbose:
        logger.info(f"Average score: {np.mean(results)}")
        logger.info(f"Minimum score: {np.min(results)}")
        logger.info(f"Maximum score: {np.max(results)}")

import numpy as np

from faraway.core.base_player import BasePlayer
from faraway.core.data_structures import MainCard, MainCardsSeries


class HumanPlayer(BasePlayer):
    def __init__(self, n_rounds: int, n_cards_hand: int = 3, use_bonus_cards: bool = True):
        super().__init__(n_rounds, n_cards_hand, use_bonus_cards)

    def evaluate_cards(
        self,
        possible_cards: np.ndarray,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if games_indices is None:
            games_indices = range(self.get_current_batch_size())
        elif isinstance(games_indices, slice):
            # Convert slice to range
            start, stop, step = games_indices.indices(self.get_current_batch_size())
            games_indices = range(start, stop, step)
        index = np.zeros(len(games_indices), dtype=np.int32)
        probabilities = np.zeros((len(games_indices), possible_cards.shape[1]))
        selected_cards = np.zeros((len(games_indices), MainCard.length()))
        for game_index in games_indices:
            # Show the cards to the human player
            print(
                f"Game #{game_index}. Round #{round_index}. Mode {mode}.\n"
                f"Possible cards: {MainCardsSeries.from_numpy(possible_cards[game_index])}"
            )
            # Ask the human player to select a card
            answer = "None"
            while eval(answer) not in range(possible_cards.shape[1]):
                try:
                    answer = input(f"Select a card index [0-{possible_cards.shape[1]-1}]: ")
                except ValueError:
                    pass
            index[game_index] = int(answer)
            probabilities[game_index, index[game_index]] = 1.0
            selected_cards[game_index] = possible_cards[game_index, index[game_index], :]
        return probabilities, index, selected_cards

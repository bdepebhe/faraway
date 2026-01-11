import torch

from faraway.torch.nn_player import BaseNNPlayer


class RandomPlayer(BaseNNPlayer):
    def __init__(
        self,
        device: torch.device | None = None,
    ):
        super().__init__(n_rounds=8, device=device)

    def evaluate_cards(
        self,
        possible_cards_tensor: torch.Tensor,
        round_index: int,
        mode: str = "play",
        games_indices: slice | range | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate possible cards and select one.

        Args:
            possible_cards_tensor: Tensor of shape (batch, n_cards, card_length)
            round_index: Current round index
            mode: "play", "draft", or "bonus"
            games_indices: Optional slice or indices to select specific batch elements from fields.
                           If None, uses all batch elements. Use slice(i, i+1) for single element.
        """
        # for random model, just use uniform logits
        logits = (
            torch.ones(
                possible_cards_tensor.shape[0], possible_cards_tensor.shape[1], device=self.device
            )
            / possible_cards_tensor.shape[1]
        )
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
        self.cards_hand_index_to_replace = index
        return probabilities, index, selected_cards

    def dump(self, path: str) -> None:
        pass

    @classmethod
    def load(cls, path: str) -> "RandomPlayer":
        return cls()

    def reset_model(self) -> None:
        pass

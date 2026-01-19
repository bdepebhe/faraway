"""
Tests for pure tensor scoring implementation.

Validates that the tensor-based scoring matches the Python/Pydantic implementation
for various game scenarios.
"""

import pytest
import torch

from faraway.core.data_structures import BonusCard, MainCard
from faraway.core.final_count import final_count
from faraway.core.player_field import PlayerField
from faraway.torch.base_game import (
    final_count_from_tensor_field_legacy,
    final_count_tensor_batched,
)
from tests.conftest import (
    IDEAL_BONUS_CARDS,
    IDEAL_GAME_SCORE,
    IDEAL_MAIN_CARDS,
)


def cards_to_tensor(
    main_cards: list[MainCard], bonus_cards: list[BonusCard]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert card lists to tensor format."""
    main_tensor = torch.stack(
        [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
    )
    # Bonus cards need to be converted to MainCard format (with id=99)
    bonus_as_main = [MainCard(**card.model_dump(), id=99) for card in bonus_cards]
    bonus_tensor = torch.stack(
        [torch.tensor(card.flatten(), dtype=torch.float32) for card in bonus_as_main]
    )
    return main_tensor, bonus_tensor


class TestIdealGameScoring:
    """Test the ideal game scenario from test_manual_max.py."""

    @pytest.fixture
    def ideal_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return cards_to_tensor(IDEAL_MAIN_CARDS, IDEAL_BONUS_CARDS)

    def test_legacy_tensor_matches_python(
        self, ideal_tensors: tuple[torch.Tensor, torch.Tensor], ideal_player_field: PlayerField
    ) -> None:
        """Legacy tensor scoring should match Python implementation."""
        main_tensor, bonus_tensor = ideal_tensors
        legacy_score = final_count_from_tensor_field_legacy(main_tensor, bonus_tensor)
        python_score = final_count(ideal_player_field)
        assert legacy_score == python_score == IDEAL_GAME_SCORE

    def test_batched_tensor_matches_python(
        self,
        ideal_tensors: tuple[torch.Tensor, torch.Tensor],
        ideal_player_field: PlayerField,
    ) -> None:
        """Batched tensor scoring should match Python implementation."""
        main_tensor, bonus_tensor = ideal_tensors
        # Add batch dimension
        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)
        tensor_score = final_count_tensor_batched(main_batched, bonus_batched)
        python_score = final_count(ideal_player_field)
        assert tensor_score.item() == python_score == IDEAL_GAME_SCORE


class TestSimpleScenarios:
    """Test simple scoring scenarios to validate individual components."""

    def test_single_card_no_bonus_flat_reward(self) -> None:
        """Single card with flat reward, no prerequisites."""
        main_cards = [
            MainCard(id=1, rewards={"flat": 5}),
        ]

        main_tensor, _ = cards_to_tensor(main_cards, [BonusCard()])
        # Create empty bonus tensor with correct shape
        bonus_tensor = torch.zeros(1, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # flat reward = 5 * 1 = 5
        assert score.item() == 5

    def test_single_card_color_reward(self) -> None:
        """Single card with color-based reward."""
        main_cards = [
            MainCard(id=1, assets={"red": 2, "green": 1}, rewards={"red": 3}),
        ]

        main_tensor = torch.stack(
            [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
        )
        bonus_tensor = torch.zeros(1, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # red reward = 3 * 2 (red assets) = 6
        assert score.item() == 6

    def test_all_4_colors_reward(self) -> None:
        """Test all_4_colors reward calculation."""
        main_cards = [
            MainCard(
                id=1,
                assets={"red": 3, "green": 2, "blue": 4, "yellow": 1},
                rewards={"all_4_colors": 5},
            ),
        ]

        main_tensor = torch.stack(
            [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
        )
        bonus_tensor = torch.zeros(1, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # all_4_colors = min(3, 2, 4, 1) = 1, reward = 5 * 1 = 5
        assert score.item() == 5

    def test_prerequisites_not_met(self) -> None:
        """Card with unmet prerequisites should score 0."""
        main_cards = [
            MainCard(id=1, prerequisites={"animal": 5}, rewards={"flat": 10}),
        ]

        main_tensor = torch.stack(
            [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
        )
        bonus_tensor = torch.zeros(1, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # Prerequisites not met (need 5 animals, have 0), score = 0
        assert score.item() == 0

    def test_prerequisites_met_by_later_cards(self) -> None:
        """Prerequisites should be checked against cards from position to end."""
        # Card 1 (played first, evaluated last): needs 2 animals, has reward
        # Card 2 (played second, evaluated first): provides 2 animals
        main_cards = [
            MainCard(id=1, prerequisites={"animal": 2}, rewards={"flat": 10}),
            MainCard(id=2, assets={"animal": 2}),
        ]

        main_tensor = torch.stack(
            [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
        )
        bonus_tensor = torch.zeros(1, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # Card 1: sees card 1 + card 2 assets = 2 animals, prereqs met, score = 10
        # Card 2: no reward
        assert score.item() == 10

    def test_bonus_cards_contribute_to_prerequisites(self) -> None:
        """Bonus cards should contribute assets for prerequisite checking."""
        main_cards = [
            MainCard(id=1, prerequisites={"animal": 2}, rewards={"flat": 10}),
        ]
        bonus_cards = [
            BonusCard(assets={"animal": 2}),
        ]

        main_tensor, bonus_tensor = cards_to_tensor(main_cards, bonus_cards)
        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # Bonus provides 2 animals, prereqs met, score = 10
        assert score.item() == 10


class TestBatchedScoring:
    """Test batched scoring with multiple games."""

    def test_batch_of_identical_games(self) -> None:
        """Batch of identical games should have identical scores."""
        main_tensor, bonus_tensor = cards_to_tensor(IDEAL_MAIN_CARDS, IDEAL_BONUS_CARDS)

        batch_size = 4
        main_batched = main_tensor.unsqueeze(0).expand(batch_size, -1, -1).clone()
        bonus_batched = bonus_tensor.unsqueeze(0).expand(batch_size, -1, -1).clone()

        scores = final_count_tensor_batched(main_batched, bonus_batched)

        assert scores.shape == (batch_size,)
        assert (scores == IDEAL_GAME_SCORE).all()

    def test_batch_of_different_games(self) -> None:
        """Batch with different games should have different scores."""
        # Game 1: ideal game (score 199)
        main1, bonus1 = cards_to_tensor(IDEAL_MAIN_CARDS, IDEAL_BONUS_CARDS)

        # Game 2: simple game with one card
        main2 = torch.zeros(8, 24, dtype=torch.float32)
        main2[0, 0] = 1  # id
        main2[0, 20] = 5  # flat reward
        bonus2 = torch.zeros(7, 24, dtype=torch.float32)

        main_batched = torch.stack([main1, main2])
        bonus_batched = torch.stack([bonus1, bonus2])

        scores = final_count_tensor_batched(main_batched, bonus_batched)

        assert scores.shape == (2,)
        assert scores[0].item() == IDEAL_GAME_SCORE
        assert scores[1].item() == 5


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_bonus_cards(self) -> None:
        """Game with no bonus cards should still score correctly."""
        main_cards = [
            MainCard(id=1, rewards={"flat": 10}),
        ]

        main_tensor = torch.stack(
            [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_cards]
        )
        # All zeros = no bonus cards
        bonus_tensor = torch.zeros(7, 24, dtype=torch.float32)

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        assert score.item() == 10

    def test_bonus_card_with_reward(self) -> None:
        """Bonus cards with rewards should contribute to score."""
        main_cards = [
            MainCard(id=1, assets={"red": 3}),
        ]
        bonus_cards = [
            BonusCard(rewards={"red": 2}),  # 2 points per red
        ]

        main_tensor, bonus_tensor = cards_to_tensor(main_cards, bonus_cards)
        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # Main card: no reward
        # Bonus card: 2 * 3 (red assets) = 6
        assert score.item() == 6

    def test_multiple_bonus_cards_with_rewards(self) -> None:
        """Multiple bonus cards should all contribute rewards."""
        main_cards = [
            MainCard(id=1, assets={"red": 2, "blue": 3}),
        ]
        bonus_cards = [
            BonusCard(rewards={"red": 1}),  # 1 * 2 = 2
            BonusCard(rewards={"blue": 2}),  # 2 * 3 = 6
        ]

        main_tensor, bonus_tensor = cards_to_tensor(main_cards, bonus_cards)
        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)

        score = final_count_tensor_batched(main_batched, bonus_batched)
        # Bonus 1: 1 * 2 = 2
        # Bonus 2: 2 * 3 = 6
        # Total: 8
        assert score.item() == 8


class TestComprehensiveComparison:
    """Compare tensor implementation against Python for many random scenarios."""

    @pytest.mark.parametrize("seed", range(10))
    def test_random_games_match_python(self, seed: int) -> None:
        """Random games should produce identical scores in both implementations."""
        torch.manual_seed(seed)

        # Generate random main cards
        n_rounds = 8
        main_cards = []
        for i in range(n_rounds):
            card = MainCard(
                id=i + 1,
                assets={
                    "rock": int(torch.randint(0, 3, (1,)).item()),
                    "animal": int(torch.randint(0, 3, (1,)).item()),
                    "vegetal": int(torch.randint(0, 3, (1,)).item()),
                    "red": int(torch.randint(0, 3, (1,)).item()),
                    "green": int(torch.randint(0, 3, (1,)).item()),
                    "blue": int(torch.randint(0, 3, (1,)).item()),
                    "yellow": int(torch.randint(0, 3, (1,)).item()),
                    "night": int(torch.randint(0, 2, (1,)).item()),
                    "map": int(torch.randint(0, 2, (1,)).item()),
                },
                prerequisites={
                    "rock": int(torch.randint(0, 2, (1,)).item()),
                    "animal": int(torch.randint(0, 3, (1,)).item()),
                    "vegetal": int(torch.randint(0, 3, (1,)).item()),
                },
                rewards={
                    "rock": int(torch.randint(0, 3, (1,)).item()),
                    "animal": int(torch.randint(0, 3, (1,)).item()),
                    "vegetal": int(torch.randint(0, 3, (1,)).item()),
                    "red": int(torch.randint(0, 3, (1,)).item()),
                    "green": int(torch.randint(0, 3, (1,)).item()),
                    "blue": int(torch.randint(0, 3, (1,)).item()),
                    "yellow": int(torch.randint(0, 3, (1,)).item()),
                    "night": int(torch.randint(0, 3, (1,)).item()),
                    "map": int(torch.randint(0, 3, (1,)).item()),
                    "all_4_colors": int(torch.randint(0, 5, (1,)).item()),
                    "flat": int(torch.randint(0, 10, (1,)).item()),
                },
            )
            main_cards.append(card)

        # Generate random bonus cards
        n_bonus = int(torch.randint(0, 7, (1,)).item())
        bonus_cards: list[BonusCard] = []
        for _ in range(n_bonus):
            bonus_card = BonusCard(
                assets={
                    "rock": int(torch.randint(0, 2, (1,)).item()),
                    "animal": int(torch.randint(0, 2, (1,)).item()),
                    "vegetal": int(torch.randint(0, 2, (1,)).item()),
                    "red": int(torch.randint(0, 2, (1,)).item()),
                    "green": int(torch.randint(0, 2, (1,)).item()),
                    "blue": int(torch.randint(0, 2, (1,)).item()),
                    "yellow": int(torch.randint(0, 2, (1,)).item()),
                    "night": int(torch.randint(0, 2, (1,)).item()),
                    "map": int(torch.randint(0, 2, (1,)).item()),
                },
                rewards={
                    "rock": int(torch.randint(0, 2, (1,)).item()),
                    "animal": int(torch.randint(0, 2, (1,)).item()),
                    "vegetal": int(torch.randint(0, 2, (1,)).item()),
                    "red": int(torch.randint(0, 2, (1,)).item()),
                    "green": int(torch.randint(0, 2, (1,)).item()),
                    "blue": int(torch.randint(0, 2, (1,)).item()),
                    "yellow": int(torch.randint(0, 2, (1,)).item()),
                    "night": int(torch.randint(0, 2, (1,)).item()),
                    "map": int(torch.randint(0, 2, (1,)).item()),
                    "all_4_colors": int(torch.randint(0, 3, (1,)).item()),
                    "flat": int(torch.randint(0, 5, (1,)).item()),
                },
            )
            bonus_cards.append(bonus_card)

        # Pad bonus cards to fixed size
        while len(bonus_cards) < 7:
            bonus_cards.append(BonusCard())  # Empty bonus card

        # Compute Python score
        player_field = PlayerField(main_cards=main_cards, bonus_cards=bonus_cards[:n_bonus])
        python_score = final_count(player_field)

        # Compute tensor score
        main_tensor, bonus_tensor = cards_to_tensor(main_cards, bonus_cards)
        # Mark empty bonus cards with id=0
        for i in range(n_bonus, 7):
            bonus_tensor[i, 0] = 0  # id = 0 means empty

        main_batched = main_tensor.unsqueeze(0)
        bonus_batched = bonus_tensor.unsqueeze(0)
        tensor_score = final_count_tensor_batched(main_batched, bonus_batched)

        assert (
            tensor_score.item() == python_score
        ), f"Seed {seed}: tensor={tensor_score.item()}, python={python_score}"


class TestDeviceCompatibility:
    """Test that scoring works on different devices."""

    def test_cpu_scoring(self) -> None:
        """Scoring should work on CPU."""
        main_tensor, bonus_tensor = cards_to_tensor(IDEAL_MAIN_CARDS, IDEAL_BONUS_CARDS)
        main_batched = main_tensor.unsqueeze(0).to("cpu")
        bonus_batched = bonus_tensor.unsqueeze(0).to("cpu")

        score = final_count_tensor_batched(main_batched, bonus_batched)
        assert score.device.type == "cpu"
        assert score.item() == IDEAL_GAME_SCORE

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_scoring(self) -> None:
        """Scoring should work on CUDA."""
        main_tensor, bonus_tensor = cards_to_tensor(IDEAL_MAIN_CARDS, IDEAL_BONUS_CARDS)
        main_batched = main_tensor.unsqueeze(0).to("cuda")
        bonus_batched = bonus_tensor.unsqueeze(0).to("cuda")

        score = final_count_tensor_batched(main_batched, bonus_batched)
        assert score.device.type == "cuda"
        assert score.item() == IDEAL_GAME_SCORE

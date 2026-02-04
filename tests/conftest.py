"""
Shared pytest fixtures for all tests.
"""

import pytest

from faraway.core.data_structures import BonusCard, MainCard
from faraway.core.player_field import PlayerField

# =============================================================================
# Ideal game data (manually optimized for maximum score of 199)
# =============================================================================

IDEAL_MAIN_CARDS = [
    MainCard(**{"id": 18, "assets": {"green": 1, "animal": 1}, "rewards": {"all_4_colors": 10}}),
    MainCard(
        **{
            "id": 23,
            "assets": {"red": 1, "rock": 1, "animal": 1, "night": 1},
            "rewards": {"all_4_colors": 10},
        }
    ),
    MainCard(
        **{
            "id": 35,
            "assets": {"yellow": 1, "night": 1, "animal": 1},
            "rewards": {"all_4_colors": 10},
        }
    ),
    MainCard(**{"id": 43, "assets": {"blue": 1, "rock": 1}, "rewards": {"all_4_colors": 10}}),
    MainCard(
        **{
            "id": 45,
            "assets": {"green": 1, "rock": 1},
            "prerequisites": {"animal": 3},
            "rewards": {"flat": 13},
        }
    ),
    MainCard(
        **{
            "id": 53,
            "assets": {"yellow": 1, "animal": 1},
            "prerequisites": {"vegetal": 2},
            "rewards": {"red": 4},
        }
    ),
    MainCard(
        **{
            "id": 63,
            "assets": {"green": 1, "map": 1},
            "prerequisites": {"animal": 2, "vegetal": 1},
            "rewards": {"flat": 15},
        }
    ),
    MainCard(
        **{
            "id": 67,
            "assets": {"green": 1, "map": 1},
            "prerequisites": {"animal": 2, "vegetal": 2},
            "rewards": {"flat": 19},
        }
    ),
]

IDEAL_BONUS_CARDS = [
    BonusCard(**{"assets": {"red": 1, "animal": 1}}),
    BonusCard(**{"assets": {"red": 1, "vegetal": 1}}),
    BonusCard(**{"assets": {"red": 1}, "rewards": {"red": 1}}),
    BonusCard(**{"assets": {"blue": 1, "animal": 1}}),
    BonusCard(**{"assets": {"blue": 1, "vegetal": 1}}),
    BonusCard(**{"assets": {"yellow": 1}, "rewards": {"all_4_colors": 4}}),
    BonusCard(**{"assets": {"yellow": 1}, "rewards": {"yellow": 1}}),
]

IDEAL_GAME_SCORE = 199


@pytest.fixture
def ideal_main_cards() -> list[MainCard]:
    """The 8 main cards for the ideal game (score 199)."""
    return IDEAL_MAIN_CARDS.copy()


@pytest.fixture
def ideal_bonus_cards() -> list[BonusCard]:
    """The 7 bonus cards for the ideal game (score 199)."""
    return IDEAL_BONUS_CARDS.copy()


@pytest.fixture
def ideal_player_field() -> PlayerField:
    """A PlayerField with the ideal game setup (score 199)."""
    return PlayerField(
        main_cards=IDEAL_MAIN_CARDS,
        bonus_cards=IDEAL_BONUS_CARDS,
    )

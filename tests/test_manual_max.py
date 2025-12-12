"""
Tests on the set of cards manually chosen to maximize the final count.
"""
import pytest

from faraway.data_structures import BonusCard, MainCard
from faraway.final_count import final_count
from faraway.player_field import PlayerField

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


@pytest.fixture
def ideal_player_field() -> PlayerField:
    return PlayerField(
        main_cards=IDEAL_MAIN_CARDS,
        bonus_cards=IDEAL_BONUS_CARDS,
    )


def test_validate_n_final_bonus_cards(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.validate_n_final_bonus_cards()


def test_validate_final_field(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.validate_final_field()


def test_validate_n_bonus_cards_to_draw(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.get_n_bonus_cards_to_draw() == 3


def test_manual_max(ideal_player_field: PlayerField) -> None:
    assert final_count(ideal_player_field) == 199

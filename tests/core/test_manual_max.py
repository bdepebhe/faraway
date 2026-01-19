"""
Tests on the set of cards manually chosen to maximize the final count.
"""

from faraway.core.final_count import final_count
from faraway.core.player_field import PlayerField
from tests.conftest import IDEAL_GAME_SCORE


def test_validate_n_final_bonus_cards(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.validate_n_final_bonus_cards()


def test_validate_final_field(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.validate_final_field()


def test_validate_n_bonus_cards_to_draw(ideal_player_field: PlayerField) -> None:
    assert ideal_player_field.get_n_bonus_cards_to_draw() == 3


def test_manual_max(ideal_player_field: PlayerField) -> None:
    assert final_count(ideal_player_field) == IDEAL_GAME_SCORE

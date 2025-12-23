import json

from faraway.core.data_structures import BonusCard, MainCard


def load_main_cards(path: str = "data/main_cards.json") -> list[MainCard]:
    with open(path) as f:
        return [MainCard(**card) for card in json.load(f)]


def load_bonus_cards(path: str = "data/bonus_cards.json") -> list[BonusCard]:
    with open(path) as f:
        return [BonusCard(**card) for card in json.load(f)]

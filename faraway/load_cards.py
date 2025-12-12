import json

from faraway.data_structures import BonusCard, MainCard


def load_main_cards() -> list[MainCard]:
    with open("data/main_cards.json") as f:
        return [MainCard(**card) for card in json.load(f)]


def load_bonus_cards() -> list[BonusCard]:
    with open("data/bonus_cards.json") as f:
        return [BonusCard(**card) for card in json.load(f)]

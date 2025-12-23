from collections.abc import Sequence

from pydantic import BaseModel

from faraway.core.data_structures import Assets, BonusCard, Card, MainCard, SummedAssets


def sum_assets(cards: Sequence[Card]) -> SummedAssets:
    summed_assets = {}
    for key in Assets.model_fields:
        summed_assets[key] = sum(getattr(card.assets, key) for card in cards)
    all_4_colors = min(
        summed_assets["red"],
        summed_assets["green"],
        summed_assets["blue"],
        summed_assets["yellow"],
    )
    return SummedAssets(**summed_assets, all_4_colors=all_4_colors)


class PlayerField(BaseModel):
    main_cards: list[MainCard] = []
    bonus_cards: list[BonusCard] = []
    n_rounds: int = 8

    def get_summed_assets(self) -> SummedAssets:
        return sum_assets([*self.main_cards, *self.bonus_cards])

    def get_n_bonus_cards_to_draw(self) -> int:
        """
        Count the number of bonus cards that the player needs to draw now.
        """
        if len(self.main_cards) >= 2 and self.main_cards[-1].id > self.main_cards[-2].id:
            return self.get_summed_assets().map + 1
        return 0

    def get_n_bonus_cards_gained(self) -> int:
        """
        Count the number of bonus cards that the player has gained by playing the main cards.
        """
        n_bonus_cards_gained = 0
        for i in range(len(self.main_cards) - 1):
            if self.main_cards[i + 1].id > self.main_cards[i].id:
                n_bonus_cards_gained += 1
        return n_bonus_cards_gained

    def validate_n_final_bonus_cards(self) -> bool:
        """
        Validate the number of final bonus cards that the player has gained.
        """
        return self.get_n_bonus_cards_gained() == len(self.bonus_cards)

    def validate_final_field(self, use_bonus_cards: bool = True) -> bool:
        """
        Validate the final field.
        """
        if use_bonus_cards:
            return len(self.main_cards) == self.n_rounds and self.validate_n_final_bonus_cards()
        else:
            return len(self.main_cards) == self.n_rounds

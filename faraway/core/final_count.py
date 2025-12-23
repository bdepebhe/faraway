from faraway.core.data_structures import (
    Prerequisites,
    Rewards,
    SummedAssets,
)
from faraway.core.player_field import PlayerField


def validate_prerequisites(prerequisites: Prerequisites, assets: SummedAssets) -> bool:
    return all(getattr(assets, key) >= value for key, value in prerequisites.model_dump().items())


def compute_value(rewards: Rewards, assets: SummedAssets) -> int:
    return sum(getattr(assets, key) * reward for key, reward in rewards.model_dump().items())


def final_count(field: PlayerField) -> int:
    total_reward = 0

    active_field = PlayerField(
        main_cards=[],
        bonus_cards=field.bonus_cards,
    )

    # count the main cards from the last to the first
    for card in field.main_cards[::-1]:
        active_field.main_cards.append(card)
        summed_assets = active_field.get_summed_assets()
        if validate_prerequisites(card.prerequisites, summed_assets):
            total_reward += compute_value(card.rewards, summed_assets)

    # count the bonus cards
    for bonus_card in field.bonus_cards:
        total_reward += compute_value(bonus_card.rewards, summed_assets)

    return total_reward

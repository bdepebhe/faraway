from pydantic import BaseModel


class Assets(BaseModel):
    red: int = 0
    green: int = 0
    blue: int = 0
    yellow: int = 0
    rock: int = 0
    animal: int = 0
    vegetal: int = 0
    night: int = 0
    map: int = 0


class Prerequisites(Assets):
    all_4_colors: int = 0


class Rewards(Prerequisites):
    flat: int = 0


class SummedAssets(Prerequisites):
    flat: int = 1


class Card(BaseModel):
    assets: Assets = Assets()
    rewards: Rewards = Rewards()


class BonusCard(Card):
    pass


class MainCard(Card):
    id: int
    prerequisites: Prerequisites = Prerequisites()

import numpy as np
from pydantic import BaseModel


class Prerequisites(BaseModel):
    rock: int = 0
    animal: int = 0
    vegetal: int = 0

    def flatten(self) -> np.ndarray:
        return np.array([value for value in self.model_dump().values()])

    @property
    def length(self) -> int:
        return len(self.model_dump())


class Assets(Prerequisites):
    red: int = 0
    green: int = 0
    blue: int = 0
    yellow: int = 0
    night: int = 0
    map: int = 0


class Rewards(Assets):
    all_4_colors: int = 0
    flat: int = 0


class SummedAssets(Assets):
    all_4_colors: int = 0
    flat: int = 1


class Card(BaseModel):
    assets: Assets = Assets()
    rewards: Rewards = Rewards()


class BonusCard(Card):
    def flatten(self) -> np.ndarray:
        flatten_assets = self.assets.flatten()
        flatten_rewards = self.rewards.flatten()
        return np.concatenate([flatten_assets, flatten_rewards])

    @property
    def length(self) -> int:
        return self.assets.length + self.rewards.length


class MainCard(Card):
    id: int = 0
    prerequisites: Prerequisites = Prerequisites()

    def flatten(self) -> np.ndarray:
        flatten_assets = self.assets.flatten()
        flatten_rewards = self.rewards.flatten()
        flatten_prerequisites = self.prerequisites.flatten()
        return np.concatenate([[self.id], flatten_assets, flatten_rewards, flatten_prerequisites])

    @property
    def length(self) -> int:
        return self.assets.length + self.rewards.length + self.prerequisites.length + 1

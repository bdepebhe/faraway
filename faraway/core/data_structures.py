import numpy as np
from pydantic import BaseModel


class Prerequisites(BaseModel):
    rock: int = 0
    animal: int = 0
    vegetal: int = 0

    def flatten(self) -> np.ndarray:
        return np.array([value for value in self.model_dump().values()])

    @classmethod
    def length(cls) -> int:
        return len(cls.model_fields)

    @classmethod
    def from_numpy(cls, numpy_array: np.ndarray) -> "Prerequisites":
        field_names = list(cls.model_fields.keys())
        return cls(**{name: int(numpy_array[i]) for i, name in enumerate(field_names)})


SHORT_PARAMS_NAMES = {
    "rock": "K",
    "animal": "A",
    "vegetal": "V",
    "red": "R",
    "green": "G",
    "blue": "B",
    "yellow": "Y",
    "night": "N",
    "map": "M",
    "all_4_colors": "4C",
    "flat": "F",
}


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

    @classmethod
    def length(cls) -> int:
        return Assets.length() + Rewards.length()

    @classmethod
    def from_main_card(cls, main_card: "MainCard") -> "BonusCard":
        # validate that prerequisites are all 0
        if not all(value == 0 for value in main_card.prerequisites.model_dump().values()):
            raise ValueError(
                f"Prerequisites are not all 0. It doesnt seem to be a bonus card. "
                f"Prerequisites: {main_card.prerequisites}"
            )
        # if main_card.id != 99:
        #     raise ValueError(f"Id is not 99. It doesnt seem to be a bonus card.)
        return cls(assets=main_card.assets, rewards=main_card.rewards)


class MainCard(Card):
    id: int
    prerequisites: Prerequisites = Prerequisites()

    def flatten(self) -> np.ndarray:
        flatten_assets = self.assets.flatten()
        flatten_rewards = self.rewards.flatten()
        flatten_prerequisites = self.prerequisites.flatten()
        return np.concatenate([[self.id], flatten_assets, flatten_rewards, flatten_prerequisites])

    @classmethod
    def length(cls) -> int:
        return Assets.length() + Rewards.length() + Prerequisites.length() + 1

    @classmethod
    def from_numpy(cls, numpy_array: np.ndarray) -> "MainCard":
        # validate the numpy array shape
        if numpy_array.shape != (cls.length(),):
            raise ValueError(
                f"Np array shape {numpy_array.shape} does not match expected shape {cls.length()}"
            )
        id = numpy_array[0]
        position = 1
        assets = Assets.from_numpy(numpy_array[position : position + Assets.length()])
        position += Assets.length()
        rewards = Rewards.from_numpy(numpy_array[position : position + Rewards.length()])
        position += Rewards.length()
        prerequisites = Prerequisites.from_numpy(
            numpy_array[position : position + Prerequisites.length()]
        )
        return cls(id=id, assets=assets, rewards=rewards, prerequisites=prerequisites)

    @classmethod
    def get_field_index(cls, field_name: str, section: str = "assets") -> int:
        """Get the index of a field in the card, after flattening the card."""
        offset = 1  # for id
        if section == "assets":
            return offset + list(Assets.model_fields.keys()).index(field_name)
        elif section == "rewards":
            offset += Assets.length()
            return offset + list(Rewards.model_fields.keys()).index(field_name)
        elif section == "prerequisites":
            offset += Assets.length() + Rewards.length()
            return offset + list(Prerequisites.model_fields.keys()).index(field_name)
        else:
            raise ValueError(f"Unknown section: {section}")

    def __str__(self) -> str:
        result = f"{self.id}/"
        for asset, value in self.assets.model_dump().items():
            if value > 0:
                result += f"{SHORT_PARAMS_NAMES[asset]}"
            if value > 1:
                result += f"{value}"
        result += "/"
        for prerequisite, value in self.prerequisites.model_dump().items():
            if value > 0:
                result += f"{SHORT_PARAMS_NAMES[prerequisite]}"
            if value > 1:
                result += f"{value}"
        result += "/"
        for reward, value in self.rewards.model_dump().items():
            if value > 0:
                result += f"{SHORT_PARAMS_NAMES[reward]}"
            if value > 1:
                result += f"{value}"
        return result


class MainCardsSeries:
    def __init__(self, list_of_main_cards: list[MainCard]):
        self.list_of_main_cards = list_of_main_cards

    @classmethod
    def from_numpy(cls, numpy_array: np.ndarray) -> "MainCardsSeries":
        list_of_main_cards = [MainCard.from_numpy(card) for card in numpy_array]
        return cls(list_of_main_cards)

    def __str__(self) -> str:
        return f"[{' '.join(f'{card}' for card in self.list_of_main_cards)}]"

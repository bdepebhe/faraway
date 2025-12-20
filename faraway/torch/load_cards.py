import torch
from loguru import logger

from faraway.core.data_structures import MainCard
from faraway.core.load_cards import load_bonus_cards, load_main_cards


def load_main_deck_tensor() -> torch.Tensor:
    # load the pydantic deck
    main_deck = load_main_cards()
    # convert to tensor
    main_deck_tensor = torch.stack(
        [torch.tensor(card.flatten(), dtype=torch.float32) for card in main_deck]
    )
    logger.info(
        f"Loaded {len(main_deck)} main cards into a tensor of shape {main_deck_tensor.shape}"
    )
    return main_deck_tensor


def load_bonus_deck_tensor() -> torch.Tensor:
    # load the pydantic deck
    bonus_deck = load_bonus_cards()
    # cast into maincard features to have the same indices as the main cards
    # NOTE: bonus cards have no id. We use -1 to indicate them to the model
    bonus_deck_as_main_cards = [MainCard(**card.model_dump(), id=-1) for card in bonus_deck]
    # convert to tensor
    bonus_deck_tensor = torch.stack(
        [torch.tensor(card.flatten(), dtype=torch.float32) for card in bonus_deck_as_main_cards]
    )
    logger.info(
        f"Loaded {len(bonus_deck)} bonus cards into a tensor of shape {bonus_deck_tensor.shape}"
    )
    return bonus_deck_tensor


if __name__ == "__main__":
    main_deck_tensor = load_main_deck_tensor()
    bonus_deck_tensor = load_bonus_deck_tensor()
    logger.info(f"Main deck tensor shape: {main_deck_tensor.shape}")
    logger.info(f"Bonus deck tensor shape: {bonus_deck_tensor.shape}")

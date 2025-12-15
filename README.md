# Faraway

![Coverage](assets/coverage-badge.svg)

A Python simulation for the [Faraway](https://boardgamegeek.com/boardgame/369875/faraway) board game.

## About the Game

Faraway is a card-drafting game where players explore mysterious lands by playing region cards. The twist? Cards are scored in **reverse order** — the last card played is scored first, meaning earlier cards can provide assets for later-scored cards.

## Features

- **Data structures** for cards, assets, prerequisites, and rewards
- **Scoring engine** that handles the reverse-order scoring mechanic
- **Card database** with main cards and bonus cards from the game (not pushed for IP reasons)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast dependency management.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and create virtual environment
uv sync

# Or install in editable mode
uv pip install -e .
```

Requires Python 3.10+ and [Pydantic](https://docs.pydantic.dev/) v2.

## Usage

```python
from faraway.data_structures import MainCard, BonusCard, Assets, Rewards, Prerequisites
from faraway.player_field import PlayerField
from faraway.final_count import final_count

# Create a player's field with their cards (in play order)
field = PlayerField(
    main_cards=[
        MainCard(id=1, assets=Assets(red=1, rock=1)),
        MainCard(id=2, assets=Assets(green=1, animal=1)),
        MainCard(
            id=3,
            assets=Assets(blue=1),
            prerequisites=Prerequisites(animal=1),
            rewards=Rewards(flat=10)
        ),
        MainCard(
            id=4,
            assets=Assets(green=1, animal=1)
        ),
    ],
    bonus_cards=[
        BonusCard(assets=Assets(yellow=1, map=1)),
    ]
)

# Calculate the final score
score = final_count(field)
print(f"Final score: {score}")
```

## Project Structure

```
faraway/
├── faraway/
│   ├── data_structures.py  # Card, Assets, Rewards, Prerequisites models
│   ├── count_utils.py      # Scoring utility functions
│   └── final_count.py      # Main scoring logic
├── data/
│   ├── main_cards.py       # Main card definitions
│   └── bonus_cards.py      # Bonus card definitions
└── pyproject.toml
```

## How Scoring Works

1. Bonus cards are always active and contribute their assets
2. Main cards are revealed and scored from **last to first**
3. For each main card:
   - Check if prerequisites are met (using assets from already-revealed cards + bonus cards)
   - If met, compute rewards based on current assets
4. Finally, bonus card rewards are added to the total

## Development

Install dev dependencies:

```bash
uv sync --extra dev
```

Set up pre-commit hooks:

```bash
pre-commit install
```

This will run the following checks on every commit:
- **Ruff** — linting and auto-fixing
- **Ruff format** — code formatting
- **Mypy** — static type checking

Run checks manually:

```bash
ruff check .          # Lint
ruff format .         # Format
mypy faraway/         # Type check
```

## License

TBD

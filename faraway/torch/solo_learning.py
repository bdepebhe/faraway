"""
Backward compatibility: re-export LearningRunner as SoloLearningGame and delegate CLI.

Prefer importing from faraway.torch.learning_runner (LearningRunner) for new code.
"""

import typer

from faraway.torch.learning_runner import LearningRunner
from faraway.torch.learning_runner import main as _main

# Backward compatibility: same class, generic name
SoloLearningGame = LearningRunner

if __name__ == "__main__":
    typer.run(_main)

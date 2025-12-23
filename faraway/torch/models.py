"""NN models for the Faraway game."""

import torch.nn as nn


def create_mlp_model(
    input_size: int,
    hidden_layers_sizes: list[int] = [512, 512],  # noqa: B006
    dropout_rate: float = 0.1,
) -> nn.Module:
    layers_sizes = [input_size] + hidden_layers_sizes
    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(layers_sizes[-1], 1))
    return nn.Sequential(*layers)

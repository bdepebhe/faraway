"""NN models for the Faraway game."""

import torch
import torch.nn as nn


def create_mlp_model(
    input_size: int,
    model_path: str | None = None,
    hidden_layers_sizes: list[int] = [512, 512],  # noqa: B006
    dropout_rate: float = 0.1,
) -> nn.Module:
    if model_path is not None:
        model = torch.load(model_path)
        if model.input_size != input_size:
            raise ValueError(
                f"Input size of model {model_path} {model.input_size} "
                f"does not match expected input size {input_size}"
            )
        return model
    layers_sizes = [input_size] + hidden_layers_sizes
    layers = []
    for i in range(len(layers_sizes) - 1):
        layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
    layers.append(nn.Linear(layers_sizes[-1], 1))
    return nn.Sequential(*layers)

"""Small PyTorch regressor used as a second model in the forecast ensemble."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class DeepLearningResult:
    predicted_return: float
    validation_mae: float
    validation_directional_accuracy: float


class _ReturnNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Tanh(),
            nn.Linear(12, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def predict_return(
    features: pd.DataFrame,
    target: pd.Series,
    inference_features: pd.DataFrame,
) -> DeepLearningResult:
    """Train a CPU-only neural regressor without using future observations."""
    if len(features) < 40:
        raise ValueError("At least 40 rows are required for deep-learning training.")

    values = features.to_numpy(dtype=np.float32)
    targets = target.to_numpy(dtype=np.float32).reshape(-1, 1)
    split = max(30, int(len(values) * 0.8))
    if split >= len(values):
        split = len(values) - 1

    train_values = values[:split]
    train_targets = targets[:split]
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_values = (values - mean) / scale
    normalized_inference = (
        inference_features.to_numpy(dtype=np.float32) - mean
    ) / scale

    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = _ReturnNetwork(values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    train_inputs = torch.from_numpy(normalized_values[:split])
    train_labels = torch.from_numpy(train_targets)

    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        loss = loss_function(model(train_inputs), train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        validation_prediction = (
            model(torch.from_numpy(normalized_values[split:])).squeeze(1).numpy()
        )
    validation_actual = targets[split:, 0]
    validation_mae = float(np.mean(np.abs(validation_actual - validation_prediction)))
    validation_directional_accuracy = float(
        np.mean(np.sign(validation_actual) == np.sign(validation_prediction))
    )

    model = _ReturnNetwork(values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    model.train()
    all_inputs = torch.from_numpy(normalized_values)
    all_labels = torch.from_numpy(targets)
    for _ in range(35):
        optimizer.zero_grad()
        loss = loss_function(model(all_inputs), all_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted_return = float(model(torch.from_numpy(normalized_inference)).item())

    return DeepLearningResult(
        predicted_return=predicted_return,
        validation_mae=validation_mae,
        validation_directional_accuracy=validation_directional_accuracy,
    )

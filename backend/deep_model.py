"""Small CPU neural regressor with chronological validation and local RNG state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import torch
from torch import nn


@dataclass(frozen=True)
class DeepLearningResult:
    predicted_return: float
    validation_mae: float
    validation_directional_accuracy: float
    validation_predictions: np.ndarray


def validation_split(rows: int) -> int:
    if rows < 40:
        raise ValueError("At least 40 rows are required for model training.")
    return max(30, int(rows * 0.8))


class _LocalLinear(nn.Linear):
    def reset_parameters(self) -> None:
        # Linear's default initialization consumes the process-wide RNG.
        # Actual initialization follows construction, using a private generator.
        pass


class _ReturnNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            _LocalLinear(input_size, 24),
            nn.Tanh(),
            _LocalLinear(24, 12),
            nn.Tanh(),
            _LocalLinear(12, 1),
        )
        generator = torch.Generator(device="cpu").manual_seed(42)
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    bound = 1 / math.sqrt(layer.in_features)
                    layer.weight.uniform_(-bound, bound, generator=generator)
                    layer.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def _fit_predict(
    values: np.ndarray, targets: np.ndarray, inference: np.ndarray
) -> np.ndarray:
    # Both feature and target scaling are fit solely on this training window.
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    target_mean = float(targets.mean())
    target_scale = float(targets.std())
    if target_scale < 1e-8:
        return np.full(len(inference), target_mean, dtype=np.float64)
    inputs = torch.from_numpy((values - mean) / scale)
    labels = torch.from_numpy((targets - target_mean) / target_scale)
    model = _ReturnNetwork(values.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = nn.SmoothL1Loss()
    model.train()
    for _ in range(35):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(inputs), labels)
        if not torch.isfinite(loss):
            raise ValueError("Neural training produced a non-finite loss.")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy((inference - mean) / scale))
        result = prediction.squeeze(1).numpy().astype(np.float64)
    result = result * target_scale + target_mean
    if not np.isfinite(result).all():
        raise ValueError("Neural inference produced a non-finite prediction.")
    return result


def predict_return(
    features: pd.DataFrame, target: pd.Series, inference_features: pd.DataFrame
) -> DeepLearningResult:
    """Validate chronologically, then refit preprocessing and weights on all rows.

    Inputs must be ordered observations with aligned next-day targets. Thread
    counts are process configuration and are never changed by this function.
    """
    split = validation_split(len(features))
    if not features.index.equals(target.index):
        raise ValueError("Features and targets must have matching row indices.")
    if features.columns.has_duplicates or not len(features.columns):
        raise ValueError("Feature columns must be nonempty and unique.")
    if not features.columns.equals(inference_features.columns):
        raise ValueError("Inference columns must match training columns in order.")
    if len(inference_features) != 1:
        raise ValueError("Exactly one inference row is required.")
    values = features.to_numpy(dtype=np.float32)
    targets = target.to_numpy(dtype=np.float32).reshape(-1, 1)
    inference = inference_features.to_numpy(dtype=np.float32)
    if not all(np.isfinite(array).all() for array in (values, targets, inference)):
        raise ValueError("Neural model inputs must be finite numbers.")
    validation_prediction = _fit_predict(
        values[:split], targets[:split], values[split:]
    )
    actual = targets[split:, 0]
    return DeepLearningResult(
        predicted_return=float(_fit_predict(values, targets, inference)[0]),
        validation_mae=float(np.mean(np.abs(actual - validation_prediction))),
        validation_directional_accuracy=float(
            np.mean(np.sign(actual) == np.sign(validation_prediction))
        ),
        validation_predictions=validation_prediction,
    )

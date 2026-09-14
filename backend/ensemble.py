"""Shared chronological blend selection for live and walk-forward forecasts."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlendResult:
    deep_weight: float
    evaluation_start: int
    predictions: np.ndarray


def select_blend(
    actual: np.ndarray, statistical: np.ndarray, deep: np.ndarray
) -> BlendResult:
    """Select on the first half of holdout; reserve the rest for evaluation.

    A small grid caps neural exposure at 30%. Ties prefer the statistical model.
    Evaluation rows never influence weight selection.
    """
    actual, statistical, deep = [
        np.asarray(x, dtype=float) for x in (actual, statistical, deep)
    ]
    if (
        actual.ndim != 1
        or actual.shape != statistical.shape
        or actual.shape != deep.shape
    ):
        raise ValueError("Blend inputs must be aligned one-dimensional predictions.")
    if len(actual) < 4 or not all(
        np.isfinite(x).all() for x in (actual, statistical, deep)
    ):
        raise ValueError("Blend selection requires at least four finite observations.")
    boundary = len(actual) // 2
    weights = np.array([0.0, 0.1, 0.2, 0.3])
    candidates = statistical[:, None] * (1 - weights) + deep[:, None] * weights
    errors = np.abs(actual[:boundary, None] - candidates[:boundary]).mean(axis=0)
    best = int(np.argmin(errors))
    return BlendResult(float(weights[best]), boundary, candidates[:, best])

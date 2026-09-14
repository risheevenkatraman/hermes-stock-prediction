from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import torch

from backend.deep_model import predict_return, validation_split
from backend.ensemble import select_blend


def inputs():
    rng = np.random.default_rng(9)
    features = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    target = pd.Series(features["a"] * 0.01 + rng.normal(0, 0.001, 60))
    return features, target, features.iloc[[-1]]


def test_neural_predictions_are_repeatable_without_global_mutation():
    x, y, latest = inputs()
    state = torch.random.get_rng_state().clone()
    threads = torch.get_num_threads()
    first = predict_return(x, y, latest)
    second = predict_return(x, y, latest)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.get_num_threads() == threads
    assert first.predicted_return == second.predicted_return
    np.testing.assert_array_equal(
        first.validation_predictions, second.validation_predictions
    )
    assert np.isfinite(first.predicted_return)


def test_concurrent_predictions_are_repeatable():
    args = inputs()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: predict_return(*args), range(2)))
    assert results[0].predicted_return == results[1].predicted_return


def test_holdout_targets_do_not_change_validation_predictions():
    x, y, latest = inputs()
    before = predict_return(x, y, latest)
    y.iloc[validation_split(len(x)) :] += 0.1
    after = predict_return(x, y, latest)
    np.testing.assert_array_equal(
        before.validation_predictions, after.validation_predictions
    )
    assert before.predicted_return != after.predicted_return


def test_constant_targets_produce_constant_predictions():
    x, y, latest = inputs()
    y[:] = 0.012
    result = predict_return(x, y, latest)
    assert result.predicted_return == pytest.approx(0.012)
    assert result.validation_mae == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("invalid", ["columns", "index", "nan", "rows"])
def test_invalid_inputs_are_rejected(invalid):
    x, y, latest = inputs()
    if invalid == "columns":
        latest = latest[list("cba")]
    elif invalid == "index":
        y.index = y.index + 1
    elif invalid == "nan":
        x.iloc[0, 0] = np.nan
    else:
        latest = latest.iloc[:0]
    with pytest.raises(ValueError):
        predict_return(x, y, latest)


def test_blend_selects_complementary_predictions_and_reserves_evaluation():
    actual = np.zeros(10)
    statistical = np.ones(10)
    deep = np.full(10, -3.0)  # Worse alone, useful as a complement.
    result = select_blend(actual, statistical, deep)
    assert result.deep_weight > 0
    changed = actual.copy()
    changed[5:] = 100
    assert select_blend(changed, statistical, deep).deep_weight == result.deep_weight
    assert result.evaluation_start == 5


def test_blend_falls_back_when_neural_predictions_worsen_error():
    result = select_blend(np.zeros(10), np.ones(10), np.full(10, 2.0))
    assert result.deep_weight == 0
    np.testing.assert_array_equal(result.predictions, np.ones(10))

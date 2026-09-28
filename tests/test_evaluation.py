import json

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from backend import evaluation
from backend.evaluation import (
    EvaluationProtocol,
    evaluate_symbol,
    summarize_predictions,
)
from benchmarks.experiment import run_experiment
from benchmarks.shadow import shadow_forecast


@pytest.fixture
def prices():
    dates = xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-07-31")[
        :100
    ]
    rng = np.random.default_rng(123)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": rng.integers(1000, 10000, len(dates)),
        }
    )


@pytest.fixture
def protocol(prices):
    return EvaluationProtocol(
        name="test-v1",
        symbols=["AAPL"],
        min_train_rows=40,
        development_start=prices.date.iloc[70].date(),
        development_end=prices.date.iloc[-1].date(),
        holdout_start="2026-09-28",
        holdout_end="2026-12-31",
        step=20,
    )


def test_all_horizons_share_origins_and_only_observable_labels(
    prices, protocol, monkeypatch
):
    fits = []

    class Spy:
        def fit(self, features, target):
            fits.append((features.index.to_list(), target.to_list()))
            return self

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(evaluation, "_regression_model", Spy)
    result = evaluate_symbol(prices, "AAPL", protocol)
    candidate = result[result.model == "price_gbt_v1"]
    assert set(candidate.horizon) == {1, 2, 3, 4, 5}
    assert candidate.groupby("horizon").origin_date.apply(tuple).nunique() == 1
    for (_, row), (indices, targets) in zip(candidate.iterrows(), fits, strict=True):
        origin = list(prices.date.dt.date.astype(str)).index(row.origin_date)
        assert max(indices) + row.horizon <= origin
        expected = prices.close.shift(-row.horizon) / prices.close - 1
        np.testing.assert_allclose(targets, expected.iloc[indices])
        assert row.target_date <= str(protocol.development_end)


def test_future_price_changes_cannot_change_past_predictions(prices, protocol):
    original = evaluate_symbol(prices, "AAPL", protocol)
    changed = prices.copy()
    changed.loc[71:, ["close", "high", "low"]] *= 1.5
    altered = evaluate_symbol(changed, "AAPL", protocol)
    origin = str(protocol.development_start)
    first = original[original.origin_date == origin]
    other = altered[altered.origin_date == origin]
    np.testing.assert_allclose(
        first.predicted_return, other.predicted_return, equal_nan=True
    )
    assert not np.allclose(first.actual_return, other.actual_return)


def test_missing_sessions_and_incomplete_windows_fail(prices, protocol):
    with pytest.raises(ValueError, match="every exchange session"):
        evaluate_symbol(prices.drop(index=35), "AAPL", protocol)
    with pytest.raises(ValueError, match="complete development period"):
        evaluate_symbol(prices.iloc[:-1], "AAPL", protocol)
    with pytest.raises(ValueError, match="Insufficient observable"):
        evaluate_symbol(
            prices, "AAPL", protocol.model_copy(update={"min_train_rows": 252})
        )


def test_holdout_cannot_be_scored_by_development_runner(prices, protocol):
    config = protocol.model_dump(mode="json")
    config.update(
        development_end=str(prices.date.iloc[85].date()),
        holdout_start=str(prices.date.iloc[86].date()),
    )
    held_out = EvaluationProtocol.model_validate(config)
    with pytest.raises(ValueError, match="Holdout rows"):
        evaluate_symbol(prices, "AAPL", held_out)
    with pytest.raises(ValueError, match="ordered and disjoint"):
        EvaluationProtocol.model_validate(
            {**config, "holdout_start": config["development_start"]}
        )


def test_metrics_keep_direction_baseline_separate():
    rows = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "horizon": 1,
                "model": model,
                "actual_return": actual,
                "predicted_return": None if model == "always_up" else 0.0,
                "predicted_direction": 1 if model == "always_up" else 0,
            }
            for model in ["always_up", "zero_return"]
            for actual in [0.01, -0.02, 0.0]
        ]
    )
    result = summarize_predictions(rows)["pooled"]["1"]
    assert result["always_up"]["return_mae"] is None
    assert result["always_up"]["directional_accuracy"] == pytest.approx(1 / 3)
    assert result["zero_return"]["return_mae"] == pytest.approx(0.01)
    assert result["zero_return"]["directional_accuracy"] == pytest.approx(1 / 3)


def test_run_preserves_inputs_protocol_predictions_and_provenance(
    prices, protocol, tmp_path
):
    data = tmp_path / "data"
    data.mkdir()
    prices.to_csv(data / "AAPL.csv", index=False)
    config = tmp_path / "protocol.json"
    config.write_text(protocol.model_dump_json())
    first = run_experiment(config, data, tmp_path / "runs")
    second = run_experiment(config, data, tmp_path / "runs")
    assert first != second
    assert (first / "inputs/AAPL.csv").read_bytes() == (data / "AAPL.csv").read_bytes()
    assert (first / "protocol.json").read_bytes() == config.read_bytes()
    assert (first / "predictions.csv").read_bytes() == (
        second / "predictions.csv"
    ).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["inputs"]["AAPL"]["sha256"]
    assert manifest["source_sha256"]["backend/evaluation.py"]
    assert (first / "source/backend/model.py").exists()
    assert "not an untouched holdout" in (first / "report.md").read_text()


def test_failed_run_is_not_reported_complete(protocol, tmp_path):
    config = tmp_path / "protocol.json"
    config.write_text(protocol.model_dump_json())
    with pytest.raises(FileNotFoundError):
        run_experiment(config, tmp_path / "missing", tmp_path / "runs")
    manifest = json.loads(next((tmp_path / "runs").glob("*/manifest.json")).read_text())
    assert manifest["status"] == "failed"


def test_shadow_saves_real_models_and_marks_historical_forecasts_ineligible(
    prices, protocol, tmp_path
):
    import joblib

    directory = shadow_forecast(prices, "AAPL", protocol, tmp_path)
    record = json.loads((directory / "record.json").read_text())
    assert record["model_name"] == "price_gbt_v1"
    assert (directory / "source/backend/model.py").is_file()
    assert record["eligible_for_prospective_scoring"] is False
    assert [row["horizon"] for row in record["forecasts"]] == [1, 2, 3, 4, 5]
    frame = evaluation.session_frame(pd.read_csv(directory / "prices.csv"))
    inference = evaluation._build_feature_frame(frame).iloc[[-1]]
    for row in record["forecasts"]:
        # Load only our own freshly generated artifact, never an untrusted pickle.
        model = joblib.load(directory / f"model-{row['horizon']}.joblib")
        assert float(model.predict(inference)[0]) == row["predicted_return"]
        assert row["training_label_end"] <= record["origin_date"]

import json
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from backend.context_features import (
    ContextSpec,
    build_context_features,
    context_returns,
)
from backend.context_model import context_metrics, evaluate_context
from backend.evaluation import EvaluationProtocol, session_frame


@pytest.fixture
def prices():
    dates = xcals.get_calendar("XNYS").sessions_in_range("2023-01-03", "2025-01-31")[
        :340
    ]
    rng = np.random.default_rng(12)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "close": close,
            "high": close + 1,
            "low": close - 1,
            "volume": rng.integers(100, 1000, len(dates)),
        }
    )


@pytest.fixture
def spec():
    return ContextSpec.model_validate_json(
        Path("benchmarks/protocols/market_context.json").read_bytes()
    )


def protocol_for(prices):
    return EvaluationProtocol(
        name="context-test",
        symbols=["AAPL"],
        development_start=prices.date.iloc[310],
        development_end=prices.date.iloc[-1],
        holdout_start="2026-09-28",
        holdout_end="2026-12-31",
        min_train_rows=40,
        step=20,
    )


def test_period_returns_use_previous_period_close(prices):
    frame = session_frame(prices)
    values = context_returns(frame)
    i = frame.index[frame.date.dt.strftime("%Y-%m-%d") == "2024-01-02"][0]
    expected = frame.close[i] / frame.close[i - 1] - 1
    assert values.loc[i, "year_to_date"] == pytest.approx(expected)
    assert values.loc[i, "quarter_to_date"] == pytest.approx(expected)
    assert values.loc[252, "return_252"] == pytest.approx(
        frame.close[252] / frame.close[0] - 1
    )


@pytest.mark.parametrize("mode", ["industry", "sector"])
def test_industry_membership_waits_for_public_availability(prices, spec, mode):
    payload = spec.model_dump(mode="json")
    payload.update(
        mode=mode,
        memberships=[
            {
                "ticker": "AAPL",
                "proxy": "TECH",
                "effective_from": prices.date.iloc[0],
                "available_at": prices.date.iloc[300] + "T23:00:00Z",
                "source": "synthetic test evidence",
            }
        ],
    )
    industry = ContextSpec.model_validate(payload)
    _, features, _, used = build_context_features(
        prices, "AAPL", {"SPY": prices, "TECH": prices}, industry
    )
    assert (
        features.loc[300, f"{mode}_return_21"] != features.loc[300, f"{mode}_return_21"]
    )
    assert pd.isna(used.iloc[300])
    assert used.iloc[301] == "TECH"
    assert features.loc[301, f"relative_{mode}_return_21"] == pytest.approx(0)
    if mode == "sector":
        assert not any("industry" in column for column in features.columns)


def test_missing_industry_is_not_market_fallback(prices, spec):
    payload = spec.model_dump(mode="json")
    payload.update(
        mode="industry",
        memberships=[
            {
                "ticker": "MSFT",
                "proxy": "TECH",
                "effective_from": "2023-01-01",
                "available_at": "2023-01-01T00:00:00Z",
                "source": "test",
            }
        ],
    )
    with pytest.raises(ValueError, match="No historical industry"):
        build_context_features(
            prices, "AAPL", {"SPY": prices}, ContextSpec.model_validate(payload)
        )


def test_future_prices_cannot_change_past_predictions(prices, spec):
    protocol = protocol_for(prices)
    original = evaluate_context(prices, "AAPL", {"SPY": prices}, spec, protocol)
    changed = prices.copy()
    changed.loc[311:, ["close", "high", "low"]] *= 1.5
    other = evaluate_context(changed, "AAPL", {"SPY": changed}, spec, protocol)
    left = original[original.origin_date == prices.date.iloc[310]]
    right = other[other.origin_date == prices.date.iloc[310]]
    np.testing.assert_allclose(left.predicted_return, right.predicted_return)
    np.testing.assert_allclose(left.up_probability, right.up_probability)
    assert not np.allclose(left.actual_return, right.actual_return)
    assert (
        original.groupby(["horizon", "model"]).origin_date.apply(tuple).nunique() == 1
    )
    assert (original.training_label_end <= original.origin_date).all()
    heads = original[original.model.isin(["ridge_price", "ridge_context"])]
    np.testing.assert_array_equal(
        heads.predicted_direction, np.where(heads.up_probability >= 0.5, 1, -1)
    )


def test_down_direction_and_brier_are_distinct_from_return_error():
    rows = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "horizon": 1,
                "model": "ridge_context",
                "actual_return": -0.02,
                "predicted_return": -0.01,
                "predicted_direction": -1,
                "up_probability": 0.2,
            }
        ]
    )
    scores = context_metrics(rows)["pooled"]["1"]["ridge_context"]
    assert scores["directional_accuracy"] == 1
    assert scores["binary_direction_accuracy"] == 1
    assert scores["return_mae"] == pytest.approx(0.01)
    assert scores["brier_score"] == pytest.approx(0.04)


def test_experiment_freezes_dataset_and_source(prices, spec, tmp_path):
    import hashlib

    from benchmarks.context import run

    inputs = tmp_path / "input"
    inputs.mkdir()
    for symbol in ("AAPL", "SPY"):
        prices.iloc[::-1].to_csv(inputs / f"{symbol}.csv", index=False)
    protocol = tmp_path / "protocol.json"
    protocol.write_text(protocol_for(prices).model_dump_json())
    context = tmp_path / "context.json"
    context.write_text(spec.model_dump_json())
    directory = run(inputs, protocol, context, tmp_path / "output")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert manifest["status"] == "complete"
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    exported = pd.read_csv(directory / "AAPL-features.csv")
    assert exported.date.tolist() == prices.date.tolist()
    assert not list(directory.rglob("*.md"))


def test_holdout_proxy_and_missing_sessions_rejected(prices, spec):
    protocol = protocol_for(prices)
    payload = protocol.model_dump(mode="json")
    payload.update(
        development_end=prices.date.iloc[-3], holdout_start=prices.date.iloc[-2]
    )
    with pytest.raises(ValueError, match="Holdout"):
        evaluate_context(
            prices.iloc[:-2],
            "AAPL",
            {"SPY": prices},
            spec,
            EvaluationProtocol.model_validate(payload),
        )
    with pytest.raises(ValueError, match="every exchange session"):
        build_context_features(prices, "AAPL", {"SPY": prices.drop(index=100)}, spec)


def test_context_spec_rejects_ambiguous_or_naive_memberships(spec):
    payload = json.loads(spec.model_dump_json())
    payload.update(
        mode="industry",
        memberships=[
            {
                "ticker": "AAPL",
                "proxy": "TECH",
                "effective_from": "2023-01-01",
                "available_at": "2023-01-01",
                "source": "test",
            }
        ],
    )
    with pytest.raises(ValueError, match="timezone"):
        ContextSpec.model_validate(payload)
    payload["memberships"][0]["available_at"] += "T00:00:00Z"
    payload["memberships"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        ContextSpec.model_validate(payload)


def test_fixed_proxy_requires_explicit_research_mode(spec):
    payload = spec.model_dump(mode="json")
    payload["fixed_sector_proxies"] = {"AAPL": "XLK"}
    with pytest.raises(ValueError, match="only permitted"):
        ContextSpec.model_validate(payload)
    payload["mode"] = "fixed_sector"
    assert ContextSpec.model_validate(payload).memberships == []
    payload["memberships"] = [
        {
            "ticker": "AAPL",
            "proxy": "XLK",
            "effective_from": "2023-01-01",
            "available_at": "2023-01-01T00:00:00Z",
            "source": "test",
        }
    ]
    with pytest.raises(ValueError, match="no dated memberships"):
        ContextSpec.model_validate(payload)


def test_sector_comparison_is_paired_and_excludes_sector_from_market_head(prices, spec):
    payload = spec.model_dump(mode="json")
    payload.update(mode="fixed_sector", fixed_sector_proxies={"AAPL": "XLK"})
    fixed = ContextSpec.model_validate(payload)
    proxies = {"SPY": prices, "XLK": prices}
    original = evaluate_context(prices, "AAPL", proxies, fixed, protocol_for(prices))
    assert {"ridge_price", "ridge_market", "ridge_sector"}.issubset(set(original.model))
    for _, group in original.groupby(["origin_date", "horizon"]):
        assert group.training_rows.nunique() == 1
        assert group.training_label_end.nunique() == 1
        assert group.actual_return.nunique() == 1
    assert (
        original.groupby(["model", "horizon"]).origin_date.apply(tuple).nunique() == 1
    )
    assert (original.context_mode == "fixed_sector").all()
    assert (original.sector_proxy == "XLK").all()
    changed = prices.copy()
    factor = np.exp(np.linspace(0, 0.5, len(changed)))
    changed[["close", "high", "low"]] = changed[["close", "high", "low"]].mul(
        factor, axis=0
    )
    altered = evaluate_context(
        prices, "AAPL", {"SPY": prices, "XLK": changed}, fixed, protocol_for(prices)
    )
    for name in ["ridge_price", "ridge_market", "price_gbt_v1"]:
        np.testing.assert_allclose(
            original.loc[original.model == name, "predicted_return"],
            altered.loc[altered.model == name, "predicted_return"],
        )
    assert not np.allclose(
        original.loc[original.model == "ridge_sector", "predicted_return"],
        altered.loc[altered.model == "ridge_sector", "predicted_return"],
    )
    future = prices.copy()
    future.loc[311:, ["close", "high", "low"]] *= 1.5
    later = evaluate_context(
        prices, "AAPL", {"SPY": prices, "XLK": future}, fixed, protocol_for(prices)
    )
    mask = original.origin_date == prices.date.iloc[310]
    np.testing.assert_allclose(
        original.loc[mask, "predicted_return"], later.loc[mask, "predicted_return"]
    )
    np.testing.assert_allclose(
        original.loc[mask, "up_probability"], later.loc[mask, "up_probability"]
    )


def test_fixed_sector_cannot_silently_use_market(prices, spec):
    payload = spec.model_dump(mode="json")
    payload.update(mode="fixed_sector", fixed_sector_proxies={"MSFT": "XLK"})
    with pytest.raises(ValueError, match="Missing fixed sector"):
        build_context_features(
            prices, "AAPL", {"SPY": prices}, ContextSpec.model_validate(payload)
        )
    payload["fixed_sector_proxies"] = {"AAPL": "XLK"}
    with pytest.raises(ValueError, match="Missing proxy prices"):
        build_context_features(
            prices, "AAPL", {"SPY": prices}, ContextSpec.model_validate(payload)
        )


def test_inner_folds_purge_label_overlap_with_session_indices():
    from backend.compact_model import inner_splits

    indices = np.arange(0, 200, 2)
    for train, validation in inner_splits(indices, 5):
        assert indices[train[-1]] + 5 <= indices[validation[0]]
        assert train[-1] < validation[0]
        assert len(validation) == 15
    with pytest.raises(ValueError, match="ordered"):
        inner_splits(indices[::-1], 5)


def test_compact_selector_does_not_use_future_outcomes(prices, spec):
    prices = pd.concat([prices, prices.iloc[-60:]], ignore_index=True)
    prices["date"] = (
        xcals.get_calendar("XNYS")
        .sessions_in_range("2023-01-03", "2024-12-31")[: len(prices)]
        .strftime("%Y-%m-%d")
    )
    protocol = protocol_for(prices).model_copy(
        update={"development_start": pd.Timestamp(prices.date.iloc[370]).date()}
    )
    payload = spec.model_dump(mode="json")
    payload.update(mode="fixed_sector", fixed_sector_proxies={"AAPL": "XLK"})
    fixed = ContextSpec.model_validate(payload)
    original = evaluate_context(
        prices,
        "AAPL",
        {"SPY": prices, "XLK": prices},
        fixed,
        protocol,
        compact_selection=True,
    )
    changed = prices.copy()
    changed.loc[371:, ["close", "high", "low"]] *= 1.4
    altered = evaluate_context(
        changed,
        "AAPL",
        {"SPY": changed, "XLK": changed},
        fixed,
        protocol,
        compact_selection=True,
    )
    mask = (original.model == "compact_selected") & (
        original.origin_date == prices.date.iloc[370]
    )
    np.testing.assert_allclose(
        original.loc[mask, "predicted_return"], altered.loc[mask, "predicted_return"]
    )
    np.testing.assert_allclose(
        original.loc[mask, "up_probability"], altered.loc[mask, "up_probability"]
    )
    assert (
        original.loc[mask, "selection"].tolist()
        == altered.loc[mask, "selection"].tolist()
    )
    for row in original.loc[original.model == "compact_selected"].itertuples():
        audit = json.loads(row.selection)
        origin = prices.index[prices.date == row.origin_date][0]
        for fold in audit["folds"]:
            assert fold["fit_label_end_index"] <= fold["validation_start_index"]
            assert fold["validation_label_end_index"] <= origin


def test_zero_targets_prefer_simple_baselines(prices, spec):
    from backend.compact_model import select_compact

    payload = spec.model_dump(mode="json")
    payload.update(mode="fixed_sector", fixed_sector_proxies={"AAPL": "XLK"})
    _, features, _, _ = build_context_features(
        prices,
        "AAPL",
        {"SPY": prices, "XLK": prices},
        ContextSpec.model_validate(payload),
    )
    x = features.iloc[252:332]
    y = pd.Series(0.0, index=x.index)
    predicted, probability, audit = select_compact(x, y, features.iloc[[333]], 1)
    assert predicted == 0 and probability == 0
    assert audit["regression"]["selected_group"] == "zero"
    assert audit["classification"]["selected_group"] == "prior"


def test_rolling_blocks_freeze_models_and_keep_labels_before_assessment(prices, spec):
    from backend.rolling_model import RollingProtocol, evaluate_rolling

    config = protocol_for(prices).model_dump(mode="json")
    config.update(
        development_start=prices.date.iloc[320],
        step=1,
        validation_rows=15,
        assessment_rows=10,
    )
    protocol = RollingProtocol.model_validate(config)
    payload = spec.model_dump(mode="json")
    payload.update(mode="fixed_sector", fixed_sector_proxies={"AAPL": "XLK"})
    fixed = ContextSpec.model_validate(payload)
    original = evaluate_rolling(
        prices, "AAPL", {"SPY": prices, "XLK": prices}, fixed, protocol
    )
    assert original.groupby("block").origin_date.nunique().tolist() == [10, 5]
    assert (original.training_label_end < original.block).all()
    assert (
        original.groupby(["block", "horizon"]).training_label_end.nunique().eq(1).all()
    )
    assert (
        original.groupby(["model", "horizon"]).origin_date.apply(tuple).nunique() == 1
    )
    audits = original.attrs["selection_audit"]
    for a in audits:
        assert a["fit_label_end"] <= a["validation_start"]
        assert a["validation_label_end"] < a["block"]
        assert a["refit_label_end"] < a["block"]
    changed = prices.copy()
    changed.loc[321:, ["close", "high", "low"]] *= 1.5
    altered = evaluate_rolling(
        changed, "AAPL", {"SPY": changed, "XLK": changed}, fixed, protocol
    )
    assert audits[:5] == altered.attrs["selection_audit"][:5]
    first = original.origin_date == prices.date.iloc[320]
    np.testing.assert_allclose(
        original.loc[first, "predicted_return"], altered.loc[first, "predicted_return"]
    )
    np.testing.assert_allclose(
        original.loc[first, "up_probability"], altered.loc[first, "up_probability"]
    )
    bad = config.copy()
    bad.update(
        development_end=prices.date.iloc[335], holdout_start=prices.date.iloc[336]
    )
    with pytest.raises(ValueError, match="Holdout"):
        evaluate_rolling(
            prices,
            "AAPL",
            {"SPY": prices, "XLK": prices},
            fixed,
            RollingProtocol.model_validate(bad),
        )


def test_rolling_rejects_insufficient_initial_training():
    from backend.rolling_model import RollingProtocol, block_indices

    p = RollingProtocol.model_validate_json(
        Path("benchmarks/protocols/sector_rolling_v1.json").read_bytes()
    )
    train, validation, known = block_indices(np.arange(252, 1500), 1200, 5, p)
    assert len(train) >= 756 and len(validation) == 126
    assert train[-1] + 5 <= validation[0]
    assert known[-1] + 5 < 1200
    with pytest.raises(ValueError, match="initial training"):
        block_indices(np.arange(252, 1500), 900, 5, p)

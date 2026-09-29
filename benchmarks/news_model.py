"""Fixed news/price research comparison; requires declared, mature phase dates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

from backend.compact_model import compact_groups
from backend.context_features import build_context_features
from backend.news import utc_timestamp
from backend.news_features import session_closes
from backend.research_heads import NEWS_NUMERIC, ResearchHead
from benchmarks.news_assemble import at_cutoff

VARIANTS = ("price_sector_only", "news_only", "price_sector_plus_news")


def observed_features(frame, events, indices):
    closes = session_closes(frame)
    rows = []
    for index in indices:
        cutoff = closes[index]
        selected = at_cutoff(events, cutoff.isoformat())
        articles = [r["article"] for r in selected]
        if not articles:
            rows.append([0.0] * len(NEWS_NUMERIC) + [""])
            continue
        weights = np.array(
            [
                a["relevance"]
                * np.exp(
                    -(cutoff - utc_timestamp(a["published_at"])).total_seconds() / 86400
                )
                for a in articles
            ]
        )
        tone = np.array([a["sentiment"] for a in articles])
        mean = np.average(tone, weights=weights)
        rows.append(
            [
                len(articles),
                mean,
                np.average(tone > 0.15, weights=weights),
                np.average(tone < -0.15, weights=weights),
                np.sqrt(np.average((tone - mean) ** 2, weights=weights)),
                " ".join(a["title"] + " " + a["summary"] for a in articles)[:50000],
            ]
        )
    return pd.DataFrame(rows, index=indices, columns=NEWS_NUMERIC + ["article_text"])


def evaluate_news(prices, ticker, proxies, spec, events, phases, protocol):
    """Return matched forecasts for validation and untouched assessment.

    Callers supply dates frozen by coverage monitoring, not chosen from outcomes.
    This function enforces phase sizes/order/embargo, complete features, matured
    labels and per-phase news coverage. It never selects model parameters.
    """
    if ticker not in protocol["symbols"]:
        raise ValueError("Ticker outside protocol")
    frame, features, _, _ = build_context_features(prices, ticker, proxies, spec)
    dates = [str(d.date()) for d in frame.date]
    indices = {}
    previous_end = None
    max_h = max(protocol["horizons"])
    for name in ("fit", "validation", "assessment"):
        values = phases[name]
        if len(values) != protocol[name + "_sessions"] or values != sorted(set(values)):
            raise ValueError(
                "Phase dates must be ordered, unique and match the protocol"
            )
        if any(d < protocol["eligible_origin_start"] for d in values):
            raise ValueError("Origins precede the eligible news experiment")
        try:
            positions = [dates.index(d) for d in values]
        except ValueError as error:
            raise ValueError("Price snapshot lacks a declared origin") from error
        if positions[-1] + max_h >= len(frame):
            raise ValueError("Phase labels have not matured")
        if previous_end is not None and positions[0] <= previous_end + max(
            max_h, protocol["boundary_embargo_sessions"]
        ):
            raise ValueError("Phase boundary lacks the required label embargo")
        protected = protocol["protected_outcome_window"]
        if any(
            dates[i] <= protected["end"] and dates[i + max_h] >= protected["start"]
            for i in positions
        ):
            raise ValueError("Labels overlap protected outcomes")
        indices[name] = positions
        previous_end = positions[-1]
    all_indices = [i for rows in indices.values() for i in rows]
    ticker_events = [r for r in events if r["article"]["ticker"] == ticker]
    x = features.loc[all_indices, compact_groups()["sector"]].join(
        observed_features(frame, ticker_events, all_indices)
    )
    if x.isna().any().any():
        raise ValueError("Incomplete research features")
    for positions in indices.values():
        if (
            float((x.loc[positions, "news_count"] > 0).mean())
            < protocol["minimum_news_fraction_per_ticker_per_phase"]
        ):
            raise ValueError("Insufficient candidate-news coverage")
    predictions = []
    for h in protocol["horizons"]:
        target = frame.close.shift(-h) / frame.close - 1
        for phase, fitting in (
            ("validation", indices["fit"]),
            ("assessment", indices["fit"] + indices["validation"]),
        ):
            evaluation = indices[phase]
            y = target.loc[fitting]
            prior, mean = float((y > 0).mean()), float(y.mean())
            outputs = {}
            for variant in VARIANTS:
                classifier = ResearchHead(variant).fit(x.loc[fitting], y)
                regressor = ResearchHead(variant, classification=False).fit(
                    x.loc[fitting], y
                )
                outputs[variant] = (
                    classifier.predict(x.loc[evaluation]),
                    regressor.predict(x.loc[evaluation]),
                )
            outputs["historical_class_prior"] = (
                np.full(len(evaluation), prior),
                np.full(len(evaluation), mean),
            )
            outputs["always_up"] = (
                np.ones(len(evaluation)),
                np.full(len(evaluation), np.nan),
            )
            outputs["zero_return"] = (
                np.full(len(evaluation), np.nan),
                np.zeros(len(evaluation)),
            )
            outputs["historical_mean_return"] = (
                np.full(len(evaluation), np.nan),
                np.full(len(evaluation), mean),
            )
            for model, (probabilities, returns) in outputs.items():
                for offset, origin in enumerate(evaluation):
                    predictions.append(
                        {
                            "ticker": ticker,
                            "horizon": h,
                            "phase": phase,
                            "model": model,
                            "origin_date": dates[origin],
                            "target_date": dates[origin + h],
                            "fit_label_end": dates[max(fitting) + h],
                            "probability": probabilities[offset],
                            "predicted_return": returns[offset],
                            "actual_return": float(target.iloc[origin]),
                            "news_present": bool(x.loc[origin, "news_count"] > 0),
                        }
                    )
    return pd.DataFrame(predictions)


def score_predictions(predictions):
    records = []
    for (phase, model, horizon), group in predictions.groupby(
        ["phase", "model", "horizon"]
    ):
        records.append(
            {
                "phase": phase,
                "model": model,
                "horizon": int(horizon),
                "observations": len(group),
                "brier": float(
                    brier_score_loss(group.actual_return > 0, group.probability)
                )
                if group.probability.notna().all()
                else None,
                "return_mae": float(
                    mean_absolute_error(group.actual_return, group.predicted_return)
                )
                if group.predicted_return.notna().all()
                else None,
            }
        )
    return records

"""Experimental six/twelve-month directional outlook, never a purchase ranking."""

from .context_features import build_context_features
from .research_heads import LONG_FEATURES, LONG_HORIZONS, ResearchHead, long_term_signal


def predict_long_term(prices, ticker, proxies, spec):
    frame, features, _, _ = build_context_features(prices, ticker, proxies, spec)
    origin = len(frame) - 1
    valid = features[LONG_FEATURES].notna().all(axis=1)
    if not valid.iloc[-1]:
        raise ValueError("Latest long-term features are incomplete")
    signals = []
    for horizon in LONG_HORIZONS:
        target = frame.close.shift(-horizon) / frame.close - 1
        matured = valid & (frame.index + horizon <= origin)
        if int(matured.sum()) < 756:
            raise ValueError(
                "Long-term research requires 756 matured training rows per horizon"
            )
        model = ResearchHead("long_term").fit(
            features.loc[matured], target.loc[matured]
        )
        probability = float(model.predict(features.iloc[[-1]])[0])
        signals.append(
            long_term_signal(probability, horizon, str(frame.date.iloc[-1].date()))
        )
    return {
        "ticker": ticker,
        "long_term": signals,
        "status": "experimental",
        "limitations": "Price/sector only; no company fundamentals or validated predictive edge. Requires a completed, adjusted daily snapshot. Not eligible for purchase recommendations.",
    }

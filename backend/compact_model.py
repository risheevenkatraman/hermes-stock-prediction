"""Small development-only search with purged chronological inner validation."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEARCH = {
    "ridge_alphas": [100.0, 1000.0, 10000.0],
    "logistic_cs": [0.001, 0.01, 0.1],
    "folds": 2,
    "validation_rows": 15,
    "minimum_fit_rows": 40,
    "regression_metric": "return_mae",
    "classification_metric": "brier_score",
    "ties": "first candidate; baselines first, stronger regularization first",
}


def compact_groups():
    stock = [
        "return_1d",
        "stock_return_21",
        "stock_return_63",
        "stock_volatility_21",
        "volume_change",
    ]
    market = stock + ["market_return_21", "market_return_63", "market_volatility_21"]
    return {
        "stock": stock,
        "market": market,
        "sector": market
        + [
            "relative_sector_return_21",
            "relative_sector_return_63",
            "sector_volatility_21",
        ],
    }


def inner_splits(index, horizon):
    """Indices are session positions, not row counts after dropping missing features."""
    indices = np.asarray(index, dtype=int)
    if len(indices) < 80 or np.any(np.diff(indices) <= 0) or horizon not in range(1, 6):
        raise ValueError(
            "Inner validation requires ordered session indices, >=80 rows and horizons 1–5."
        )
    folds = []
    for start in (len(indices) - 30, len(indices) - 15):
        validation = np.arange(start, start + 15)
        train = np.flatnonzero(
            (np.arange(len(indices)) < start) & (indices + horizon <= indices[start])
        )
        if len(train) < SEARCH["minimum_fit_rows"]:
            raise ValueError("Insufficient purged inner training history.")
        folds.append((train, validation))
    return folds


def predict_candidate(candidate, x, y, latest, classification):
    group, regularization = candidate
    if group == "zero":
        return np.zeros(len(latest))
    if group == "mean":
        return np.full(len(latest), float(y.mean()))
    labels = (y > 0).astype(int)
    if group == "prior" or (classification and labels.nunique() == 1):
        return np.full(len(latest), float(labels.mean()))
    columns = compact_groups()[group]
    estimator = (
        LogisticRegression(C=regularization, max_iter=1000, random_state=42)
        if classification
        else Ridge(alpha=regularization)
    )
    model = make_pipeline(StandardScaler(), estimator)
    model.fit(x[columns], labels if classification else y)
    return (
        model.predict_proba(latest[columns])[:, 1]
        if classification
        else model.predict(latest[columns])
    )


def select_compact(x, y, latest, horizon):
    if not x.index.equals(y.index):
        raise ValueError("Features and targets must align.")
    folds = inner_splits(x.index, horizon)
    audit = {
        "folds": [
            {
                "fit_rows": len(train),
                "fit_label_end_index": int(x.index[train[-1]] + horizon),
                "validation_start_index": int(x.index[validation[0]]),
                "validation_label_end_index": int(x.index[validation[-1]] + horizon),
            }
            for train, validation in folds
        ]
    }
    predictions = []
    for classification in (False, True):
        candidates = (
            [("prior", None)] if classification else [("zero", None), ("mean", None)]
        )
        grid = (
            SEARCH["logistic_cs"]
            if classification
            else sorted(SEARCH["ridge_alphas"], reverse=True)
        )
        candidates += [(group, value) for group in compact_groups() for value in grid]
        scores = []
        for candidate in candidates:
            errors = []
            for train, validation in folds:
                predicted = predict_candidate(
                    candidate,
                    x.iloc[train],
                    y.iloc[train],
                    x.iloc[validation],
                    classification,
                )
                actual = (
                    (y.iloc[validation].to_numpy() > 0).astype(int)
                    if classification
                    else y.iloc[validation].to_numpy()
                )
                errors.extend(
                    (predicted - actual) ** 2
                    if classification
                    else np.abs(predicted - actual)
                )
            scores.append(float(np.mean(errors)))
        best = int(np.argmin(scores))
        selected = candidates[best]
        predictions.append(
            float(predict_candidate(selected, x, y, latest, classification)[0])
        )
        audit["classification" if classification else "regression"] = {
            "selected_group": selected[0],
            "regularization": selected[1],
            "inner_score": scores[best],
            "candidates": [
                {"group": group, "regularization": value, "inner_score": score}
                for (group, value), score in zip(candidates, scores, strict=True)
            ],
        }
    return predictions[0], predictions[1], audit

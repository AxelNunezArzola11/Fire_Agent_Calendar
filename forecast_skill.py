"""Out-of-fold skill harness for the pseudo-labeled fire forecaster.

Measures pseudo-label recovery -- NOT true forecast skill: labels are
same-window pseudo-labels (24h activity -> 1, 7d silence -> 0), so
these metrics bound fitting quality, never predictive capability.
Manual stratified K-fold (numpy only; sklearn is not installed and
installs are out of scope). Compares XGBoost out-of-fold vs the
deterministic scorer vs a base-rate dummy on identical folds, plus a
reliability table. Deterministic (fixed seed).

Caveats (read before quoting any number):
- The deterministic scorer consumes hist_fire_count_7d, which also
  defines label 0: it is label-adjacent, a reference, not a fair
  competitor. The dummy (predict the base rate) is the fair floor.
- Weather features are contemporaneous with the label window: even
  out-of-fold scores do not estimate forward forecasting skill.
- Time-forward validation (train past weeks, score next day) remains
  future work requiring archived weather + outcome labels.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBClassifier

import forecast_engine as engine

SEED = 42
FOLDS = 5
EPS = 1e-9

# Must match forecast_engine._get_model_and_predictions exactly.
XGB_PARAMS = {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.1,
              "random_state": SEED, "eval_metric": "logloss", "verbosity": 0}


def pseudo_labels(feature_df):
    """Mirror the engine's pseudo-label rule (ambiguous rows dropped)."""
    label_1 = feature_df["hist_fire_count_24h"] > 0
    label_0 = feature_df["hist_fire_count_7d"] == 0
    idx = feature_df.index[label_1 | label_0]
    return idx, np.where(label_1[idx], 1, 0)


def stratified_folds(labels, k=FOLDS, seed=SEED):
    """Deterministic stratified fold ids (numpy only)."""
    rng = np.random.default_rng(seed)
    folds = np.zeros(len(labels), dtype=int)
    for cls in (0, 1):
        members = np.flatnonzero(labels == cls)
        rng.shuffle(members)
        folds[members] = np.arange(len(members)) % k
    return folds


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=float)
    clipped = np.clip(np.asarray(y_prob, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(clipped)
                           + (1.0 - y_true) * np.log(1.0 - clipped)))


def reliability_table(y_true, y_prob, bins=10):
    """Per-bin (lo, hi, mean_pred, frac_pos, n); empty bins kept."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 \
            else (y_prob >= lo) & (y_prob <= hi)
        sel = y_true[mask]
        rows.append({"bin_lo": round(float(lo), 2),
                     "bin_hi": round(float(hi), 2),
                     "mean_pred": float(y_prob[mask].mean())
                     if mask.sum() else None,
                     "frac_pos": float(sel.mean())
                     if len(sel) else None,
                     "n": int(mask.sum())})
    return rows


def _oof_predict(X, labels, k, seed):
    """Out-of-fold XGB probabilities (may raise ImportError: no sklearn)."""
    folds = stratified_folds(labels, k=k, seed=seed)
    oof = np.zeros(len(labels))
    for fold in range(k):
        train = folds != fold
        if train.sum() == 0 or (~train).sum() == 0:
            continue
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X[train], labels[train])
        oof[~train] = clf.predict_proba(X[~train])[:, 1]
    return oof


def evaluate(feature_df, k=FOLDS, seed=SEED) -> dict:
    """Out-of-fold pseudo-label recovery for XGB vs baselines."""
    idx, labels = pseudo_labels(feature_df)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if len(idx) < engine.MIN_LABELLED_SAMPLES or n_pos < 1 or n_neg < 1:
        return {"eligible": False,
                "reason": f"need >= {engine.MIN_LABELLED_SAMPLES} "
                          f"labelled rows with both classes "
                          f"(have {len(idx)}: {n_pos} pos, {n_neg} neg)",
                "n_labelled": int(len(idx)), "n_pos": n_pos,
                "n_neg": n_neg}
    frame = feature_df.loc[idx]
    X = frame[engine.MODEL_FEATURE_COLS].values
    det = np.array([engine.deterministic_score(row)
                    for _, row in frame.iterrows()])
    base = float(labels.mean())
    dummy = np.full(len(idx), base)
    out = {
        "eligible": True, "k": k, "seed": seed,
        "n_labelled": int(len(idx)), "n_pos": n_pos, "n_neg": n_neg,
        "deterministic": {"brier": brier_score(labels, det),
                          "logloss": log_loss_score(labels, det)},
        "dummy_base_rate": {"brier": brier_score(labels, dummy),
                            "logloss": log_loss_score(labels, dummy),
                            "base_rate": base},
        "reliability_deterministic": reliability_table(labels, det),
        "caveat": "pseudo-label recovery only; not forecast skill",
    }
    try:
        oof = _oof_predict(X, labels, k, seed)
    except ImportError:
        out["xgb_oof"] = {"status": "unavailable",
                          "reason": "xgboost requires sklearn "
                                    "(not installed)"}
    else:
        out["xgb_oof"] = {"brier": brier_score(labels, oof),
                          "logloss": log_loss_score(labels, oof)}
        out["reliability_xgb_oof"] = reliability_table(labels, oof)
    return out

"""Train-normal-only source reliability utilities for rank experiments."""

from __future__ import annotations

import numpy as np


SOURCE_PRIORS = {"rocket": 0.45, "exp55": 0.275, "exp56": 0.275}


def source_reliability(bundle, seed=20260710):
    scores = np.asarray(bundle.get("train_scores", []), dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 8:
        return 0.0, {"tail_ratio": None, "bootstrap_cv": None, "train_exceed": float(bundle.get("train_exceed_rate", 1.0))}
    q25, q50, q75, q90, q95 = np.percentile(scores, [25, 50, 75, 90, 95])
    iqr = max(float(q75 - q25), 1e-8)
    tail_ratio = max(0.0, float(q95 - q90)) / iqr
    rng = np.random.default_rng(seed + len(scores))
    boot = [np.percentile(rng.choice(scores, size=len(scores), replace=True), 95) for _ in range(16)]
    bootstrap_cv = float(np.std(boot)) / iqr
    train_exceed = float(bundle.get("train_exceed_rate", 1.0))
    reliability = 1.0 / (1.0 + tail_ratio + bootstrap_cv + min(2.0, 25.0 * max(0.0, train_exceed)))
    return float(np.clip(reliability, 0.0, 1.0)), {
        "tail_ratio": tail_ratio,
        "bootstrap_cv": bootstrap_cv,
        "train_exceed": train_exceed,
        "median": float(q50),
    }


def source_reliabilities(bundles):
    values, diagnostics = {}, {}
    for offset, name in enumerate(("rocket", "exp55", "exp56")):
        value, detail = source_reliability(bundles[{"rocket": "rocket_exp40", "exp55": "exp55_best", "exp56": "exp56_best"}[name]], 20260710 + offset)
        values[name] = value
        diagnostics[name] = detail
    return values, diagnostics


def adaptive_weights(reliabilities):
    raw = {name: SOURCE_PRIORS[name] * (0.25 + reliabilities.get(name, 0.0)) for name in SOURCE_PRIORS}
    total = sum(raw.values()) or 1.0
    return {name: value / total for name, value in raw.items()}

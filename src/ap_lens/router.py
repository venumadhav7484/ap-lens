"""Code-level routing for the decisions the agent can't see.

Run 2 showed LightGBM is usually right when it confidently predicts FIX_PRICE or FIX_QTY,
but the agent overrides it. So those calls skip the agent: a line goes straight to the
buyer queue when LightGBM's probability clears a threshold.

The threshold is chosen on the precedent pool only, never on the held-out set: 5-fold
out-of-fold LightGBM probabilities, folds grouped by case so no case is in both halves. The
pick is the lowest threshold whose out-of-fold precision reaches TARGET_PRECISION with at
least MIN_SUPPORT routed lines.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .baselines import LGBMModel

ROUTED = ("FIX_PRICE", "FIX_QTY")
TARGET_PRECISION = 0.80
MIN_SUPPORT = 5
GRID = [round(float(x), 2) for x in np.arange(0.30, 0.96, 0.05)]


def choose_threshold(pool: pd.DataFrame, folds: int = 5) -> dict:
    from sklearn.model_selection import GroupKFold
    pool = pool.reset_index(drop=True)
    probs: list[dict | None] = [None] * len(pool)
    for fit_idx, val_idx in GroupKFold(n_splits=folds).split(pool, groups=pool[C.CASE]):
        model = LGBMModel(pool.iloc[fit_idx])
        for i, p in zip(val_idx, model.proba(pool.iloc[val_idx])):
            probs[i] = p
    y = pool["label"].tolist()
    rows = []
    for t in GRID:
        picks = [(max(ROUTED, key=lambda a: p.get(a, 0)), lab) for p, lab in zip(probs, y)
                 if max(p.get(a, 0) for a in ROUTED) >= t]
        n = len(picks)
        correct = sum(a == lab for a, lab in picks)
        rows.append({"threshold": float(t), "routed": n, "correct": correct,
                     "precision": correct / n if n else None})
    ok = [r for r in rows if r["routed"] >= MIN_SUPPORT and r["precision"] is not None
          and r["precision"] >= TARGET_PRECISION]
    chosen = ok[0]["threshold"] if ok else None
    out = {"chosen_threshold": chosen, "target_precision": TARGET_PRECISION, "min_support": MIN_SUPPORT,
           "pool_points": int(len(pool)), "folds": folds, "sweep": rows,
           "note": "Chosen on out-of-fold predictions over the precedent pool; the held-out set is never used."}
    C.out("router_threshold.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def route(probs: dict[str, float], threshold: float | None) -> tuple[str, float] | None:
    """Return (action, probability) if the rule fires, else None."""
    if threshold is None:
        return None
    action = max(ROUTED, key=lambda a: probs.get(a, 0))
    p = probs.get(action, 0)
    return (action, p) if p >= threshold else None

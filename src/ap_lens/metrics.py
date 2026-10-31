"""Scoring against what people actually did next."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from . import config as C


def score(test: pd.DataFrame, pred: np.ndarray) -> dict:
    y = test["label"].to_numpy()
    pred = np.asarray(pred)
    answered = pred != "ESCALATE"
    rel = pred == "RELEASE"
    safe = rel & (y == "RELEASE") & ~test["reversed_later"].to_numpy()
    unsafe = rel & ~test["match_ok"].to_numpy()
    labels = C.AGENT_ACTIONS
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=C.ACTIONS, average="macro", zero_division=0)),
        "accuracy_when_answered": float(accuracy_score(y[answered], pred[answered])) if answered.any() else None,
        "abstain_rate": float(1 - answered.mean()),
        "release_proposals": int(rel.sum()),
        "safe_release_precision": float(safe.sum() / rel.sum()) if rel.any() else None,
        "releases_without_match": int(unsafe.sum()),
        "confusion": {"labels": labels,
                      "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }

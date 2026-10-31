"""Three non-LLM baselines the agent has to beat.

1. majority  - always the most common label in the precedent pool
2. batch     - the rule a nightly job would run: release on a clean match, wait for goods,
               otherwise send to the buyer
3. lightgbm  - gradient boosting on prefix features, trained on the precedent pool
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as C

NUM = ["po_value", "inv_value", "gr_value", "inv_vs_po_pct", "n_events", "n_inv", "n_gr",
       "n_cancel_gr", "n_chg_price", "n_chg_qty", "days_since_create"]
BOOL = ["gr_posted", "gr_since_inv", "match_ok"]


def features(df: pd.DataFrame, vendor_freq: pd.Series | None = None) -> pd.DataFrame:
    X = df[NUM].astype(float).copy()
    for b in BOOL:
        X[b] = df[b].astype(float)
    X["point_is_gr"] = (df["point_type"] == C.GR).astype(float)
    for c in C.CATEGORICAL:
        if c == "vendor":
            continue
        X[c] = df[c].astype("category")
    vf = vendor_freq if vendor_freq is not None else df["vendor"].value_counts(normalize=True)
    X["vendor_freq"] = df["vendor"].map(vf).fillna(0.0)
    return X


def majority(pool: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.array([pool["label"].mode()[0]] * len(test))


def batch_rule(test: pd.DataFrame) -> np.ndarray:
    out = []
    for _, r in test.iterrows():
        if r["match_ok"]:
            out.append("RELEASE")
        elif not r["gr_posted"]:
            out.append("WAIT_GR")
        elif r["inv_vs_po_pct"] is not None and not np.isnan(r["inv_vs_po_pct"]) and r["inv_vs_po_pct"] < 0:
            out.append("FIX_QTY")
        else:
            out.append("FIX_PRICE")
    return np.array(out)


class LGBMModel:
    """LightGBM trained on the precedent pool. Shared by the baseline and the agent's
    ml_prediction tool, so both see exactly the same model."""

    def __init__(self, pool: pd.DataFrame):
        self.vf = pool["vendor"].value_counts(normalize=True)
        Xp = features(pool, self.vf)
        self.cats = {c: Xp[c].cat.categories for c in Xp.columns if str(Xp[c].dtype) == "category"}
        self.labels = sorted(pool["label"].unique())
        y = pool["label"].map({l: i for i, l in enumerate(self.labels)})
        self.model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                        class_weight="balanced", random_state=C.SEED, verbose=-1)
        self.model.fit(Xp, y)

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        X = features(df, self.vf)
        for c, cats in self.cats.items():
            X[c] = pd.Categorical(X[c].astype(object), categories=cats)
        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.array([self.labels[i] for i in self.model.predict(self._X(df))])

    def proba(self, df: pd.DataFrame) -> list[dict[str, float]]:
        P = self.model.predict_proba(self._X(df))
        return [{l: float(p[i]) for i, l in enumerate(self.labels)} for p in P]


def lightgbm(pool: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return LGBMModel(pool).predict(test)


def run_all(df: pd.DataFrame) -> dict[str, np.ndarray]:
    pool, test = df[df["pool"]], df[df["heldout"]]
    return {"majority": majority(pool, test), "batch_rule": batch_rule(test),
            "lightgbm": lightgbm(pool, test)}

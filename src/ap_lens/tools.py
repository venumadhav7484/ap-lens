"""The four tools, shared by the in-process agent and the MCP server.

Every read is cut at the decision point (`as_of_seq`), so neither the agent nor an MCP
client can see what happened later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import decisions as D

_NN_NUM = ["inv_vs_po_pct", "n_events", "n_chg_price", "n_chg_qty", "days_since_create", "po_value"]
_NN_BOOL = ["gr_posted", "gr_since_inv", "match_ok"]
_NN_CAT = ["item_type", "spend_area_text", "document_type"]


def _resource_type(r) -> str:
    if C.is_human(r):
        return "person"
    return "none" if r in (None, "NONE") or (isinstance(r, float) and np.isnan(r)) else "batch job"


@dataclass
class CaseStore:
    events: pd.DataFrame
    cases: pd.DataFrame
    points: pd.DataFrame
    _by_case: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_disk(cls) -> "CaseStore":
        from .load import load_frames
        ev, cs = load_frames()
        return cls(ev, cs, D.load())

    def rows(self, case_id: str) -> pd.DataFrame:
        if not self._by_case:
            for cid, g in self.events.sort_values([C.CASE, C.TS, "seq"]).groupby(C.CASE, sort=False):
                self._by_case[cid] = g.reset_index(drop=True)
        return self._by_case[case_id]

    def point(self, point_id: str) -> dict:
        r = self.points.loc[self.points["point_id"] == point_id]
        if r.empty:
            raise KeyError(f"unknown decision point {point_id}")
        return r.iloc[0].to_dict()

    # ---- tools ----
    def get_case(self, point_id: str, max_events: int = 50) -> dict:
        p = self.point(point_id)
        rows = self.rows(p[C.CASE]).iloc[: p["as_of_seq"] + 1]
        t0 = rows[C.TS].iloc[0]
        attrs = self.cases.loc[self.cases[C.CASE] == p[C.CASE]].iloc[0].to_dict()
        timeline = [{"step": int(i), "activity": r[C.ACT],
                     "day": round((r[C.TS] - t0).total_seconds() / 86400, 1),
                     "by": _resource_type(r[C.RES]),
                     "value_eur": None if pd.isna(r.get(C.VALUE)) else float(r.get(C.VALUE))}
                    for i, r in rows.tail(max_events).iterrows()]
        keep = [C.ITEM_CATEGORY] + C.CATEGORICAL
        return {"decision_point": point_id, "triggered_by": p["point_type"],
                "case": {k: (None if pd.isna(attrs.get(k)) else attrs.get(k)) for k in keep if k in attrs},
                "timeline": timeline}

    def check_match(self, point_id: str) -> dict:
        return D.check_match(self.point(point_id))

    @cached_property
    def _nn(self):
        pool = self.points[self.points["pool"]].reset_index(drop=True)
        X = self._nn_matrix(pool, fit=True)
        nn = NearestNeighbors(n_neighbors=min(6, len(pool))).fit(X)
        return nn, pool

    def _nn_matrix(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        num = df[_NN_NUM].astype(float).fillna(0.0).to_numpy()
        if fit:
            self._scaler = StandardScaler().fit(num)
            self._cats = {c: sorted(df[c].dropna().astype(str).unique()) for c in _NN_CAT}
        parts = [self._scaler.transform(num), df[_NN_BOOL].astype(float).to_numpy() * 2]
        for c in _NN_CAT:
            vals = df[c].astype(str).to_numpy()
            parts.append(np.stack([(vals == v).astype(float) for v in self._cats[c]], axis=1))
        return np.hstack(parts)

    @cached_property
    def _lgbm(self):
        from .baselines import LGBMModel
        return LGBMModel(self.points[self.points["pool"]])

    def ml_prediction(self, point_id: str) -> dict:
        """LightGBM's class probabilities for this decision point, trained on the precedent pool.
        It sees vendor, item and history patterns the timeline alone doesn't show."""
        p = self.points.loc[self.points["point_id"] == point_id]
        probs = self._lgbm.proba(p)[0]
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        return {"predicted": ranked[0][0], "probabilities": {k: round(v, 3) for k, v in ranked},
                "model": "LightGBM on prefix features, trained on decisions before the cutoff",
                "how_to_use": "A statistical prior from past cases, not a rule. Weigh it against the case evidence."}

    def similar_cases(self, point_id: str, k: int = 5) -> dict:
        nn, pool = self._nn
        p = self.points.loc[self.points["point_id"] == point_id]
        _, idx = nn.kneighbors(self._nn_matrix(p), n_neighbors=min(k + 1, len(pool)))
        out = []
        for i in idx[0]:
            r = pool.iloc[i]
            if r["point_id"] == point_id or len(out) >= k:
                continue
            out.append({"case": r["point_id"], "what_happened_next": r["label"],
                        "done_by": "person" if r["label_by_human"] else "batch job",
                        "days_until_then": round(float(r["days_to_label"]), 1),
                        "item_type": r["item_type"], "spend_area": r["spend_area_text"],
                        "goods_received": bool(r["gr_posted"]), "match_ok": bool(r["match_ok"]),
                        "invoice_vs_po_pct": None if pd.isna(r["inv_vs_po_pct"]) else float(r["inv_vs_po_pct"])})
        return {"precedents": out, "note": "Cases decided before the evaluation cutoff only."}


def as_json(obj) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)

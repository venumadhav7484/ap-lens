"""Step 2: decision points, labels, features and the time split.

A decision point is a moment in the "invoice before goods receipt" flow where an invoice is
recorded and still blocked: the invoice receipt itself, or a goods receipt that lands while
the invoice is open. The label is what happened next among the decision activities in
config.LABEL_OF. Features use only the prefix up to and including the decision point.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


def _prefix_facts(rows: pd.DataFrame) -> dict:
    acts = rows[C.ACT].tolist()
    vals = rows[C.VALUE].tolist() if C.VALUE in rows else [None] * len(rows)

    def last(a):
        for act, v in zip(reversed(acts), reversed(vals)):
            if act == a and v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
        return None

    po = None
    for act, v in zip(acts, vals):
        if act == C.CREATE and v is not None and not (isinstance(v, float) and np.isnan(v)):
            po = float(v)
            break
    last_inv_i = max((i for i, a in enumerate(acts) if a == C.INV), default=-1)
    gr_since_inv = sum(1 for a in acts[last_inv_i + 1:] if a == C.GR) - \
        sum(1 for a in acts[last_inv_i + 1:] if a == C.CANCEL_GR)
    gr_net = acts.count(C.GR) - acts.count(C.CANCEL_GR)
    return {"po_value": po, "inv_value": last(C.INV), "gr_value": last(C.GR),
            "gr_posted": gr_net > 0, "gr_since_inv": gr_since_inv > 0,
            "n_events": len(acts), "n_inv": acts.count(C.INV), "n_gr": acts.count(C.GR),
            "n_cancel_gr": acts.count(C.CANCEL_GR), "n_chg_price": acts.count(C.CHG_PRICE),
            "n_chg_qty": acts.count(C.CHG_QTY)}


def check_match(f: dict) -> dict:
    """Deterministic 3-way match on the prefix. This is code, not model judgement."""
    po, inv, gr = f.get("po_value"), f.get("inv_value"), f.get("gr_value")

    def close(a, b):
        return a is not None and b is not None and abs(a - b) <= C.VALUE_TOLERANCE * max(abs(a), abs(b), 1e-9)

    def valid(x):  # an amount the matcher can trust: present, finite, positive
        return x is not None and np.isfinite(x) and x > 0

    values_known = valid(po) and valid(inv)
    inv_vs_po = None if not values_known else (inv - po) / po
    ok = (bool(f.get("gr_posted")) and values_known and close(inv, po)
          and (gr is None or (valid(gr) and close(gr, inv))))
    return {"gr_posted": bool(f.get("gr_posted")), "values_known": values_known,
            "invoice_vs_po_pct": None if inv_vs_po is None else round(100 * inv_vs_po, 2),
            "within_tolerance": ok}


def build(ev: pd.DataFrame, cs: pd.DataFrame) -> pd.DataFrame:
    flow = cs.loc[cs[C.ITEM_CATEGORY] == C.BLOCK_FLOW, C.CASE]
    e = ev[ev[C.CASE].isin(flow)].sort_values([C.CASE, C.TS, "seq"])
    attrs = cs.set_index(C.CASE)
    pts = []
    for cid, rows in e.groupby(C.CASE, sort=False):
        rows = rows.reset_index(drop=True)
        acts = rows[C.ACT].tolist()
        open_inv, taken = False, 0
        for i, a in enumerate(acts):
            if a == C.INV:
                open_inv = True
            if not open_inv or a not in (C.INV, C.GR):
                if a in (C.UNBLOCK, C.CLEAR, C.CANCEL_INV, C.DEL_ITEM):
                    open_inv = False
                continue
            nxt = next((j for j in range(i + 1, len(acts)) if acts[j] in C.LABEL_OF), None)
            if nxt is None:
                continue
            f = _prefix_facts(rows.iloc[: i + 1])
            if C.BLOCKED_ONLY and a == C.INV and f["gr_posted"]:
                continue  # goods already received: the invoice is never blocked, no decision to make
            label_act = acts[nxt]
            pts.append({
                "point_id": f"{cid}#{i}", C.CASE: cid, "as_of_seq": int(i), "point_type": a,
                "ts": rows.at[i, C.TS], "label_activity": label_act, "label": C.LABEL_OF[label_act],
                "label_ts": rows.at[nxt, C.TS], "label_resource": rows.at[nxt, C.RES],
                "label_by_human": C.is_human(rows.at[nxt, C.RES]),
                "days_to_label": (rows.at[nxt, C.TS] - rows.at[i, C.TS]).total_seconds() / 86400,
                "reversed_later": any(x in (C.CANCEL_INV, C.CHG_PRICE, C.CHG_QTY) for x in acts[nxt + 1:]),
                "days_since_create": (rows.at[i, C.TS] - rows.at[0, C.TS]).total_seconds() / 86400,
                **f, **{k: attrs.at[cid, k] if k in attrs.columns else None for k in C.CATEGORICAL},
            })
            taken += 1
            if taken >= C.MAX_POINTS_PER_CASE:
                break
    df = pd.DataFrame(pts)
    m = df.apply(lambda r: check_match(r.to_dict()), axis=1, result_type="expand")
    df["match_ok"] = m["within_tolerance"]
    df["inv_vs_po_pct"] = m["invoice_vs_po_pct"]
    return df


def split(df: pd.DataFrame) -> pd.DataFrame:
    """Precedent pool = points whose outcome is known before the cutoff. Held-out = human
    decisions after the cutoff, stratified by label, at most HELDOUT_N."""
    cutoff = df["ts"].quantile(0.8)
    df = df.copy()
    df["pool"] = df["label_ts"] < cutoff
    cand = df[(df["ts"] >= cutoff) & df["label_by_human"]]
    n = min(C.HELDOUT_N, len(cand))
    frac = n / max(len(cand), 1)
    rng = np.random.default_rng(C.SEED)
    keep = []
    for _, g in cand.groupby("label"):
        k = max(min(len(g), 15), int(round(len(g) * frac)))
        keep.extend(rng.choice(g.index.to_numpy(), size=min(k, len(g)), replace=False).tolist())
    keep = keep[:n] if len(keep) > n else keep
    df["heldout"] = df.index.isin(keep)
    df.attrs["cutoff"] = str(cutoff)
    return df


def run() -> dict:
    from .load import load_frames
    ev, cs = load_frames()
    df = split(build(ev, cs))
    df.to_parquet(C.parquet("decisions"), index=False)
    human = int(df["label_by_human"].sum())
    summary = {
        "source": C.SOURCE, "decision_points": int(len(df)), "cutoff": df.attrs["cutoff"],
        "human_decisions": human, "gate_ok": human >= 300,
        "label_counts": df["label"].value_counts().to_dict(),
        "heldout": int(df["heldout"].sum()), "pool": int(df["pool"].sum()),
        "heldout_label_counts": df.loc[df["heldout"], "label"].value_counts().to_dict(),
    }
    C.out("decisions_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return summary


def load() -> pd.DataFrame:
    return pd.read_parquet(C.parquet("decisions"))

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
            "n_chg_qty": acts.count(C.CHG_QTY), **_line_facts(rows)}


def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _line_facts(rows: pd.DataFrame) -> dict:
    """Quantity and unit price per line, for feeds that carry them (the CSV pilot feed).
    PO terms follow later Change Price / Change Quantity events; received quantity is net of
    cancelled receipts; the invoice is the latest one recorded."""
    out = {"po_qty": None, "po_unit_price": None, "received_qty": None,
           "invoiced_qty": None, "invoice_unit_price": None}
    if "qty" not in rows or "unit_price" not in rows or rows["qty"].isna().all():
        return out
    received = 0.0
    for a, q, p in zip(rows[C.ACT], rows["qty"], rows["unit_price"]):
        q, p = _num(q), _num(p)
        if a == C.CREATE:
            out["po_qty"], out["po_unit_price"] = q, p
        elif a == C.CHG_PRICE and p is not None:
            out["po_unit_price"] = p
        elif a == C.CHG_QTY and q is not None:
            out["po_qty"] = q
        elif a == C.GR and q is not None:
            received += q
        elif a == C.CANCEL_GR and q is not None:
            received -= q
        elif a == C.INV:
            out["invoiced_qty"], out["invoice_unit_price"] = q, p
    out["received_qty"] = received
    return out


def check_match(f: dict) -> dict:
    """Deterministic 3-way match on the prefix. This is code, not model judgement."""
    po, inv, gr = f.get("po_value"), f.get("inv_value"), f.get("gr_value")

    def close(a, b):
        return a is not None and b is not None and abs(a - b) <= C.VALUE_TOLERANCE * max(abs(a), abs(b), 1e-9)

    def valid(x):  # an amount the matcher can trust: present, finite, positive
        return x is not None and np.isfinite(x) and x > 0

    gr_posted = bool(f.get("gr_posted"))
    lp = {k: _num(f.get(k)) for k in ("po_qty", "po_unit_price", "received_qty", "invoiced_qty",
                                      "invoice_unit_price")}
    if all(valid(lp[k]) for k in ("po_qty", "po_unit_price", "invoiced_qty", "invoice_unit_price")):
        # Line-level match: price per unit, and invoiced quantity against what was received.
        tol = C.VALUE_TOLERANCE
        price_ok = abs(lp["invoice_unit_price"] - lp["po_unit_price"]) <= tol * lp["po_unit_price"]
        recv = lp["received_qty"] or 0.0
        qty_ok = (recv > 0 and lp["invoiced_qty"] <= recv * (1 + tol)
                  and lp["invoiced_qty"] <= lp["po_qty"] * (1 + tol))
        mismatch = ("goods not received" if recv <= 0 or not gr_posted else
                    "price" if not price_ok else "quantity" if not qty_ok else None)
        return {"gr_posted": gr_posted, "values_known": True, "line_level": True,
                "po_unit_price": lp["po_unit_price"], "invoice_unit_price": lp["invoice_unit_price"],
                "price_diff_pct": round(100 * (lp["invoice_unit_price"] / lp["po_unit_price"] - 1), 2),
                "po_qty": lp["po_qty"], "received_qty": recv, "invoiced_qty": lp["invoiced_qty"],
                "mismatch": mismatch, "within_tolerance": gr_posted and price_ok and qty_ok}

    values_known = valid(po) and valid(inv)
    inv_vs_po = None if not values_known else (inv - po) / po
    ok = (gr_posted and values_known and close(inv, po)
          and (gr is None or (valid(gr) and close(gr, inv))))
    return {"gr_posted": gr_posted, "values_known": values_known,
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
    pct = m["invoice_vs_po_pct"] if "invoice_vs_po_pct" in m else pd.Series(np.nan, index=df.index)
    if "price_diff_pct" in m:  # line-level feeds: the price difference is the comparable signal
        pct = pct.fillna(m["price_diff_pct"])
    df["inv_vs_po_pct"] = pct
    df["mismatch"] = m["mismatch"] if "mismatch" in m else None
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

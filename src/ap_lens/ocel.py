"""Adapter for a second, differently shaped log: OCEL 2.0 Procure-to-Pay (simulated).

Source: "Simulated Object-Centric Event Logs (OCEL 2.0) for Order-to-Cash, Procure-to-Pay,
Hiring, and Hospital Patient Lifecycle Processes", Zenodo record 13879980, CC BY 4.0, file
02_p2p.xml. Object-centric: events link to Purchase Order / Goods Receipt / Invoice objects.

The adapter flattens it to AP-Lens's case schema (one case per purchase order) and renames
activities to the BPIC19 names the pipeline uses. A successful three-way match counts as a
release; "Mismatch Found" counts as FIX_PRICE (the log doesn't say whether price or quantity
was off). The original activity name is kept in `activity_raw`.

Finding recorded in outputs/ocel_label_check.json: in this simulation the match outcome is
independent of the invoice-vs-PO amount, so it is used to test portability, not accuracy.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd
from lxml import etree

from . import config as C

OCEL_FILE = C.DATA / "02_p2p.xml"
RENAME = {
    "Create PO": C.CREATE, "Create PO (Batch)": C.CREATE,
    "Receive Goods": C.GR, "Receive Invoice": C.INV,
    "Three-way Match - Successful": C.UNBLOCK,
    "Three-way Match - Mismatch Found": C.CHG_PRICE,
    "Make Payment": C.CLEAR,
}


def _attrs(el) -> dict:
    a = el.find("attributes")
    return {} if a is None else {x.get("name"): x.text for x in a}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build() -> dict:
    root = etree.parse(str(OCEL_FILE)).getroot()
    objs = {o.get("id"): (o.get("type"), _attrs(o)) for o in root.find("objects")}
    # each non-PO object belongs to the PO it first appears with
    owner: dict[str, str] = {}
    events = []
    for e in root.find("events"):
        rel = [r.get("object-id") for r in (e.find("objects") if e.find("objects") is not None else [])]
        pos = [i for i in rel if objs.get(i, ("",))[0] == "Purchase Order"]
        if pos:
            for i in rel:
                owner.setdefault(i, pos[0])
        events.append((e, rel))
    rows, case_attrs = [], defaultdict(dict)
    for e, rel in events:
        po = next((i for i in rel if objs.get(i, ("",))[0] == "Purchase Order"), None) \
            or next((owner[i] for i in rel if i in owner), None)
        if po is None or e.get("type") not in RENAME:
            continue
        at = _attrs(e)
        typ = e.get("type")
        value = None
        if typ == "Receive Invoice":
            value = next((_f(objs[i][1].get("amount")) for i in rel if objs.get(i, ("",))[0] == "Invoice"), None)
        elif typ in ("Create PO", "Create PO (Batch)", "Receive Goods"):
            value = _f(objs[po][1].get("total_amount"))
        user = at.get("user") or ""
        resource = "batch_po" if typ == "Create PO (Batch)" else (f"user_{user}" if user else "NONE")
        rows.append({C.CASE: po, C.ACT: RENAME[typ], "activity_raw": typ, C.RES: resource,
                     C.TS: pd.Timestamp(e.get("time"), tz="UTC"), C.VALUE: value})
        sup = next((objs[i][1] for i in rel if objs.get(i, ("",))[0] == "Supplier"), None)
        gr = next((objs[i][1] for i in rel if objs.get(i, ("",))[0] == "Goods Receipt"), None)
        ca = case_attrs[po]
        if sup:
            ca.setdefault("vendor", sup.get("supplier_name"))
            ca.setdefault("document_type", sup.get("terms"))
            ca.setdefault("company", sup.get("supplier_rating"))
        if gr:
            ca.setdefault("item_type", f"goods {gr.get('condition', '?').lower()}")
        if at.get("cost_center"):
            ca.setdefault("spend_area_text", at.get("cost_center"))
    ev = pd.DataFrame(rows).sort_values([C.CASE, C.TS]).reset_index(drop=True)
    ev["seq"] = ev.groupby(C.CASE).cumcount()
    cs = pd.DataFrame([{C.CASE: k, C.ITEM_CATEGORY: C.BLOCK_FLOW, **v} for k, v in case_attrs.items()
                       if k in set(ev[C.CASE])])
    for col in C.CATEGORICAL + ["sub_spend_area_text"]:
        if col not in cs:
            cs[col] = None
    ev.to_parquet(C.parquet("events"), index=False)
    cs.to_parquet(C.parquet("cases"), index=False)
    check = label_check(root, objs)
    (C.OUT / "ocel_label_check.json").write_text(json.dumps(check, indent=1))
    from .load import inventory
    inv = inventory(ev, cs)
    inv["label_check"] = check
    return inv


def label_check(root, objs) -> dict:
    """Does the simulated match outcome depend on the data a matcher can see?"""
    out: dict = defaultdict(lambda: {"n": 0, "amount_differs": 0, "goods_damaged": 0})
    for e in root.find("events"):
        t = e.get("type")
        if not t.startswith("Three-way"):
            continue
        rel = [r.get("object-id") for r in e.find("objects")]
        get = lambda ty: next((objs[i][1] for i in rel if objs.get(i, ("",))[0] == ty), None)
        po, inv, gr = get("Purchase Order"), get("Invoice"), get("Goods Receipt")
        d = out[t]
        d["n"] += 1
        if po and inv and _f(po.get("total_amount")):
            if abs(_f(inv.get("amount")) - _f(po.get("total_amount"))) > C.VALUE_TOLERANCE * _f(po.get("total_amount")):
                d["amount_differs"] += 1
        if gr and gr.get("condition") == "Damaged":
            d["goods_damaged"] += 1
    res = {k: {**v, "amount_differs_share": v["amount_differs"] / v["n"], "damaged_share": v["goods_damaged"] / v["n"]}
           for k, v in out.items()}
    res["verdict"] = ("Outcome looks independent of amount and goods condition: usable for portability, "
                      "not for accuracy.")
    return res

"""Synthetic BPIC19-shaped XES for development and tests.

Same keys, activity names and flow types as the real log, random content. Every trace
carries `synthetic=true`. Nothing computed from this file is ever reported as a result.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr

from . import config as C

FLOWS = [("3-way match, invoice before GR", 0.55, False, True),
         ("3-way match, invoice after GR", 0.35, True, True),
         ("2-way match", 0.07, False, False),
         ("Consignment", 0.03, False, True)]
SPEND = ["Packaging", "Sales", "Logistics", "CAPEX & SOCS", "Labs"]
ITEM_TYPES = ["Standard", "Service", "Subcontracting", "Third-party"]
DOC_TYPES = ["Standard PO", "Framework order", "EC Purchase order"]


def _users(rng, n):
    return [f"user_{rng.randint(1, n):03d}" for _ in range(1)][0]


def _trace(rng: random.Random, i: int, t0: datetime) -> tuple[dict, list[tuple]]:
    cat, _, grbased, gr = rng.choices(FLOWS, weights=[f[1] for f in FLOWS])[0]
    vendor = f"vendor_{rng.randint(1, 120):04d}"
    attrs = {"concept:name": f"S{i:06d}_00001", "Item Category": cat, "Item Type": rng.choice(ITEM_TYPES),
             "Spend area text": rng.choice(SPEND), "Sub spend area text": rng.choice(["A", "B", "C"]),
             "Document Type": rng.choice(DOC_TYPES), "Company": rng.choice(["c_1", "c_2", "c_3"]),
             "Vendor": vendor, "GR-Based Inv. Verif.": grbased, "Goods Receipt": gr, "synthetic": True}
    po = round(rng.lognormvariate(7, 1.2), 2)
    t = t0
    ev = []

    def add(act, res, val, days):
        nonlocal t
        t = t + timedelta(days=days, hours=rng.random() * 8)
        ev.append((act, res, val, t))

    u = lambda: f"user_{rng.randint(1, 40):03d}"
    b = lambda: f"batch_{rng.randint(0, 6):02d}"
    add(C.CREATE, u(), po, 0)
    if rng.random() < 0.3:
        add("Receive Order Confirmation", u(), po, rng.uniform(0, 3))
    if cat == "3-way match, invoice before GR":
        mismatch = rng.random()
        inv = po * (1.08 if mismatch < 0.15 else 0.93 if mismatch < 0.23 else 1.0)
        add("Vendor creates invoice", "NONE", inv, rng.uniform(1, 10))
        add(C.INV, u(), inv, rng.uniform(1, 12))
        if mismatch > 0.95:
            add(C.CANCEL_INV, u(), inv, rng.uniform(1, 9))
            add(C.DEL_ITEM, u(), inv, rng.uniform(0, 3))
            return attrs, ev
        if mismatch < 0.15:
            add(C.CHG_PRICE, u(), inv, rng.uniform(2, 14))
        elif mismatch < 0.23:
            add(C.CHG_QTY, u(), inv, rng.uniform(2, 14))
        add(C.GR, u(), inv, rng.uniform(1, 20))
        if rng.random() < 0.05:
            add(C.CANCEL_GR, u(), inv, rng.uniform(0, 4))
            add(C.GR, u(), inv, rng.uniform(1, 10))
        human = rng.random() < (0.35 + 0.3 * (attrs["Item Type"] == "Service"))
        add(C.UNBLOCK, u() if human else b(), inv, rng.uniform(2, 15) if human else rng.uniform(0.5, 3))
        add(C.CLEAR, b(), inv, rng.uniform(5, 30))
    elif cat == "3-way match, invoice after GR":
        add(C.GR, u(), po, rng.uniform(2, 20))
        add("Vendor creates invoice", "NONE", po, rng.uniform(1, 10))
        add(C.INV, u(), po, rng.uniform(1, 12))
        add(C.CLEAR, b(), po, rng.uniform(5, 40))
    elif cat == "2-way match":
        add(C.INV, u(), po, rng.uniform(3, 20))
        add(C.CLEAR, b(), po, rng.uniform(5, 30))
    else:
        add(C.GR, u(), po, rng.uniform(2, 20))
    return attrs, ev


def _attr(k, v):
    tag = "boolean" if isinstance(v, bool) else "float" if isinstance(v, float) else "string"
    val = str(v).lower() if isinstance(v, bool) else str(v)
    return f'<{tag} key={quoteattr(k)} value={quoteattr(val)}/>'


def write(path: Path = C.FIXTURE_XES, n: int = 3000, seed: int = C.SEED) -> Path:
    rng = random.Random(seed)
    base = datetime(2018, 1, 1, tzinfo=timezone.utc)
    with open(path, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<log xes.version="1.0">\n')
        for i in range(n):
            attrs, ev = _trace(rng, i, base + timedelta(days=rng.uniform(0, 330)))
            f.write("<trace>" + "".join(_attr(k, v) for k, v in attrs.items()))
            for act, res, val, ts in ev:
                f.write("<event>" + _attr("concept:name", act) + _attr("org:resource", res)
                        + _attr("Cumulative net worth (EUR)", float(round(val, 2)))
                        + f'<date key="time:timestamp" value="{ts.isoformat()}"/></event>')
            f.write("</trace>\n")
        f.write("</log>\n")
    return path

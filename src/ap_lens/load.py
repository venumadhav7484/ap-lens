"""XES -> Parquet with a streaming parser (no PM4Py: it is AGPL-3.0).

Writes <source>_events.parquet (one row per event) and <source>_cases.parquet (one row per
trace, with all trace attributes). Attribute keys are normalised to snake_case.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from lxml import etree

from . import config as C

_NORM = re.compile(r"[^0-9a-z]+")


def norm(key: str) -> str:
    if key == "concept:name":
        return "name"
    return _NORM.sub("_", key.lower()).strip("_")


def _value(el) -> object:
    tag = etree.QName(el).localname
    v = el.get("value")
    if tag == "boolean":
        return v == "true"
    if tag in ("float", "int"):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    return v


def parse_xes(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    events, cases = [], []
    for _, trace in etree.iterparse(str(path), events=("end",), tag=("{*}trace", "trace")):
        attrs = {}
        evs = []
        for child in trace:
            if etree.QName(child).localname == "event":
                e = {norm(c.get("key")): _value(c) for c in child if c.get("key")}
                evs.append(e)
            elif child.get("key"):
                attrs[norm(child.get("key"))] = _value(child)
        cid = attrs.pop("name", None)
        attrs[C.CASE] = cid
        cases.append(attrs)
        for i, e in enumerate(evs):
            e[C.CASE] = cid
            e[C.ACT] = e.pop("name", None)
            e[C.TS] = e.pop("time_timestamp", None)
            e[C.RES] = e.pop("org_resource", None)
            e["seq"] = i
            events.append(e)
        trace.clear()
        while trace.getprevious() is not None:
            del trace.getparent()[0]
    ev = pd.DataFrame(events)
    ev[C.TS] = pd.to_datetime(ev[C.TS], utc=True, errors="coerce", format="mixed")
    return ev, pd.DataFrame(cases)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    """Fetch the real log once. Only run after the user has approved the download."""
    if C.XES_FILE.exists():
        print("already downloaded:", C.XES_FILE)
        return
    part = C.XES_FILE.with_suffix(".part")
    print("downloading", C.XES_URL)
    with urllib.request.urlopen(C.XES_URL, timeout=60) as r, open(part, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    got = md5(part)
    if got != C.XES_MD5:
        print(f"WARNING md5 {got} != published {C.XES_MD5}; keeping file, check before use")
    part.rename(C.XES_FILE)


def build() -> dict:
    ev, cs = parse_xes(C.xes_path())
    ev.to_parquet(C.parquet("events"), index=False)
    cs.to_parquet(C.parquet("cases"), index=False)
    return inventory(ev, cs)


def inventory(ev: pd.DataFrame, cs: pd.DataFrame) -> dict:
    """Step 0 gate: what the log really contains, so config.py can be checked against it."""
    from . import config as K
    expected = [K.CREATE, K.INV, K.GR, K.CANCEL_GR, K.CLEAR, K.UNBLOCK, K.CHG_PRICE,
                K.CHG_QTY, K.CANCEL_INV, K.DEL_ITEM]
    acts = ev[K.ACT].value_counts()
    res = ev[K.RES].fillna("NONE").astype(str)
    inv = {
        "source": K.SOURCE,
        "xes": K.XES_SOURCES[K.XES_SOURCE]["note"] if K.SOURCE == "real" else "synthetic fixture",
        "events": int(len(ev)),
        "cases": int(len(cs)),
        "activities": int(acts.size),
        "resources": int(res.nunique()),
        "human_resources": int(res[res.map(K.is_human)].nunique()),
        "resource_prefixes": res.str.extract(r"^([A-Za-z]+)")[0].value_counts().head(8).to_dict(),
        "activity_counts": {k: int(v) for k, v in acts.items()},
        "missing_expected_activities": [a for a in expected if a not in acts.index],
        "case_columns": sorted(cs.columns),
        "event_columns": sorted(ev.columns),
        "item_category_counts": cs.get(K.ITEM_CATEGORY, pd.Series(dtype=str)).value_counts().to_dict(),
        "value_present_share": float(ev[K.VALUE].notna().mean()) if K.VALUE in ev else 0.0,
        "value_varies_within_case_share": _value_variation(ev),
        "ts_range": [str(ev[K.TS].min()), str(ev[K.TS].max())],
    }
    C.out("step0_inventory.json").write_text(json.dumps(inv, indent=1, default=str))
    return inv


def _value_variation(ev: pd.DataFrame) -> float:
    if C.VALUE not in ev:
        return 0.0
    g = ev.dropna(subset=[C.VALUE]).groupby(C.CASE)[C.VALUE].nunique()
    return float((g > 1).mean()) if len(g) else 0.0


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ev = pd.read_parquet(C.parquet("events"))
    cs = pd.read_parquet(C.parquet("cases"))
    return ev, cs

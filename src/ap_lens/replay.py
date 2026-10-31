"""Step 6: the KPI, before and after, replayed on the held-out human decisions.

Before: every held-out decision was a person's touch. After: where the agent's final action
is RELEASE and a person also released, that touch is automated, and the release moves to the
moment of the decision point. All "after" numbers are simulated on a replayed public log.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from . import decisions as D


def run(model: str) -> dict:
    pts = D.load()
    pred = pd.read_parquet(C.out(f"predictions_{model}.parquet"))
    t = pts.merge(pred[["point_id", "pred"]], on="point_id")
    releases = t[t["label"] == "RELEASE"]
    auto = releases[releases["pred"] == "RELEASE"]
    n = len(t)
    saved_days = auto["days_to_label"].clip(lower=0)
    before_days = releases["days_to_label"].clip(lower=0)
    after_days = before_days.copy()
    after_days.loc[auto.index] = 0.0
    kpi = {
        "model": model, "source": C.SOURCE, "decisions": int(n),
        "human_releases": int(len(releases)),
        "releases_automated": int(len(auto)),
        "manual_touches_per_100_before": 100.0,
        "manual_touches_per_100_after": float(100 * (n - len(auto)) / n) if n else None,
        "manual_release_share_automated": float(len(auto) / len(releases)) if len(releases) else None,
        "median_days_block_to_release_before": float(before_days.median()) if len(releases) else None,
        "median_days_block_to_release_after": float(after_days.median()) if len(releases) else None,
        "days_saved_total": float(saved_days.sum()),
        "wrong_releases": int(((t["pred"] == "RELEASE") & (t["label"] != "RELEASE")).sum()),
        "note": "Simulated on a replayed public log (BPI Challenge 2019); not a client deployment.",
    }
    C.out(f"kpi_{model}.json").write_text(json.dumps(kpi, indent=1))
    return kpi

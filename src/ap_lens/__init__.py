"""AP-Lens: a forward-deployed-style case study on the BPI Challenge 2019 purchase-to-pay log."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(prog="ap-lens")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fixture", help="write the synthetic dev log")
    sub.add_parser("download", help="fetch the real log (needs approval; ~700 MB)")
    sub.add_parser("load", help="step 0: XES -> Parquet + inventory")
    sub.add_parser("discover", help="step 1: discovery SQL")
    sub.add_parser("decisions", help="step 2: decision points, labels, split")
    sub.add_parser("router", help="choose the FIX_PRICE/FIX_QTY routing threshold on the pool")
    sh = sub.add_parser("shadow", help="replay events in time order; agent decides live, audited")
    sh.add_argument("--model", default="claude-haiku-5-5")
    sh.add_argument("--n", type=int, default=100)
    sh.add_argument("--rule", action="store_true", help="decide with the batch rule instead of the LLM ($0)")
    rd = sub.add_parser("readiness", help="readiness scorecard: fuzz, failures, drift (+ live checks)")
    rd.add_argument("--model", default="claude-haiku-5-5")
    rd.add_argument("--offline", action="store_true", help="skip the live-model checks")
    rd.add_argument("--load-n", type=int, default=2000)
    rd.add_argument("--only", default=None, help="rerun one check: fuzz|failures|drift|injection|amounts|load")
    rd.add_argument("--workers", type=int, default=16)
    e = sub.add_parser("eval", help="step 5: baselines + agent + guardrail suite")
    e.add_argument("--models", default="claude-haiku-5-5")
    e.add_argument("--n", type=int, default=None)
    e.add_argument("--effort", default="low")
    e.add_argument("--cap", type=float, default=None)
    e.add_argument("--no-ml", action="store_true", help="agent without the ml_prediction tool (run 1 setup)")
    r = sub.add_parser("replay", help="step 6: KPI before/after")
    r.add_argument("--model", default="claude-haiku-5-5")
    p = sub.add_parser("report", help="HTML report + README numbers")
    p.add_argument("--model", default="claude-haiku-5-5")
    a = ap.parse_args()

    from . import config as C
    if a.cmd == "fixture":
        from . import fixture
        print(fixture.write())
    elif a.cmd == "download":
        from . import load
        load.download()
    elif a.cmd == "load":
        if C.SOURCE == "ocel":
            from . import ocel
            inv = ocel.build()
        else:
            from . import load
            inv = load.build()
        print(json.dumps({k: inv[k] for k in ("events", "cases", "activities", "human_resources",
                                              "missing_expected_activities", "item_category_counts",
                                              "value_varies_within_case_share")}, indent=1, default=str))
    elif a.cmd == "discover":
        from . import discover
        print(json.dumps(discover.run()["block_flow"], indent=1, default=str))
    elif a.cmd == "decisions":
        from . import decisions
        print(json.dumps(decisions.run(), indent=1, default=str))
    elif a.cmd == "router":
        from . import decisions, router
        df = decisions.load()
        r = router.choose_threshold(df[df["pool"]])
        print(json.dumps({k: r[k] for k in ("chosen_threshold", "target_precision", "pool_points")}, indent=1))
    elif a.cmd == "shadow":
        from . import shadow
        client = shadow.RuleDecider() if a.rule else None
        print(json.dumps(shadow.run(a.model, a.n, client=client), indent=1, default=str))
    elif a.cmd == "readiness":
        from . import readiness
        card = readiness.run(a.model, live=not a.offline, load_n=a.load_n, only=a.only, workers=a.workers)
        print(json.dumps({k: v.get("pass") for k, v in card.items() if isinstance(v, dict)}, indent=1))
    elif a.cmd == "eval":
        from . import evals
        rep = evals.run([m for m in a.models.split(",") if m], n=a.n, effort=a.effort, cap=a.cap or C.SPEND_CAP_USD,
                         use_ml=not a.no_ml)
        print(json.dumps({"baselines": {k: round(v["macro_f1"], 3) for k, v in rep["baselines"].items()},
                          "agents": {k: {"acc": round(v["accuracy"], 3), "f1": round(v["macro_f1"], 3),
                                         "unsafe": v["guardrail_suite"]["final_releases"],
                                         "cost": round(v["cost_usd_total"], 3)} for k, v in rep["agents"].items()},
                          "spend_usd": round(rep["spend_usd"], 3)}, indent=1))
    elif a.cmd == "replay":
        from . import replay
        print(json.dumps(replay.run(a.model), indent=1))
    elif a.cmd == "report":
        from . import report
        report.render(a.model)
        if C.SOURCE == "real":
            report.update_readme(a.model)
        print(C.out("report.html"))

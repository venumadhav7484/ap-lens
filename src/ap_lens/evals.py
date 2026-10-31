"""Step 5: score baselines and the agent on the held-out human decisions, run the guardrail
suite, and report cost and latency. Writes outputs/metrics.json, predictions_<model>.parquet
and traces_<model>.jsonl.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from . import baselines, config as C, guardrail_suite
from .agent import Agent
from .metrics import score
from .tools import CaseStore


def _run_agent(agent: Agent, ids: list[str], trace_path, spent: list[float], cap: float,
               workers: int = 8) -> dict[str, dict]:
    res: dict[str, dict] = {}
    with open(trace_path, "a") as tf, ThreadPoolExecutor(workers) as ex:
        futs = {}
        for pid in ids:
            if sum(spent) >= cap:
                break
            futs[ex.submit(agent.decide, pid)] = pid
        for f in as_completed(futs):
            pid = futs[f]
            try:
                out = f.result()
            except Exception as e:  # an API failure counts as an escalation, and is logged
                out = {"final": {"action": "ESCALATE", "reason": f"error: {e}"},
                       "trace": {"point_id": pid, "error": str(e), "cost_usd": 0.0}}
            c = out["trace"].get("cost_usd") or 0.0
            spent.append(0.0 if math.isnan(c) else c)
            res[pid] = out
            tf.write(json.dumps({**out["trace"], "final_decision": out["final"]}, default=str) + "\n")
    return res


def run(models: list[str], n: int | None = None, effort: str = "low", cap: float = C.SPEND_CAP_USD,
        use_ml: bool = True) -> dict:
    store = CaseStore.from_disk()
    df = store.points
    test = df[df["heldout"]].sort_values("point_id")
    if n:
        test = test.head(n)
    report: dict = {"source": C.SOURCE, "agent_tools": "4 (with ml_prediction)" if use_ml else "3", "n_heldout": int(len(test)),
                    "label_counts": test["label"].value_counts().to_dict(), "baselines": {}, "agents": {}}
    full_test = df[df["heldout"]]
    for name, pred in baselines.run_all(df).items():
        pred = pd.Series(pred, index=full_test.index).loc[test.index].to_numpy()
        report["baselines"][name] = score(test, pred)

    gstore, gids = guardrail_suite.store_with_suite(store)
    gstore._nn, gstore._lgbm  # train shared models once, before the worker threads start
    spent: list[float] = []
    for model in models:
        agent = Agent(gstore, model=model, effort=effort, use_ml=use_ml)
        tpath = C.out(f"traces_{model}.jsonl")
        tpath.write_text("")
        res = _run_agent(agent, test["point_id"].tolist(), tpath, spent, cap)
        done = test[test["point_id"].isin(res)]
        pred = np.array([res[p]["final"]["action"] for p in done["point_id"]])
        proposed = np.array([res[p]["trace"].get("proposed", "ESCALATE") for p in done["point_id"]])
        traces = [res[p]["trace"] for p in done["point_id"]]
        m = score(done, pred)
        m["proposed_before_guardrail"] = score(done, proposed)
        m["guardrail_blocked_releases"] = int(sum(t.get("guardrail_violation", False) for t in traces))
        costs = [t.get("cost_usd") or 0.0 for t in traces]
        lat = [t.get("latency_s") for t in traces if t.get("latency_s") is not None]
        m["cost_usd_total"] = float(np.nansum(costs))
        m["cost_usd_per_1000"] = float(np.nanmean(costs) * 1000) if costs else None
        m["latency_p50_s"] = float(np.percentile(lat, 50)) if lat else None
        m["latency_p95_s"] = float(np.percentile(lat, 95)) if lat else None
        m["mean_tool_calls"] = float(np.mean([len(t.get("tool_calls", [])) for t in traces])) if traces else None
        m["errors"] = int(sum(1 for t in traces if t.get("error")))
        m["completed"] = int(len(done))
        gpath = C.out(f"guardrail_traces_{model}.jsonl")
        gpath.write_text("")
        g = _run_agent(agent, gids, gpath, spent, cap + 1.0)
        m["guardrail_suite"] = {
            "cases": len(gids),
            "final_releases": int(sum(r["final"]["action"] == "RELEASE" for r in g.values())),
            "model_proposed_releases": int(sum(r["trace"].get("proposed") == "RELEASE" for r in g.values())),
            "by_pattern": _by_pattern(g),
        }
        report["agents"][model] = m
        pd.DataFrame({"point_id": done["point_id"], "label": done["label"], "pred": pred,
                      "proposed": proposed}).to_parquet(C.out(f"predictions_{model}.parquet"), index=False)
    report["spend_usd"] = float(sum(spent))
    C.out("metrics.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def _by_pattern(g: dict) -> dict:
    out: dict[str, dict] = {}
    for pid, r in g.items():
        name = pid.split("#")[0].removeprefix("GUARD_").rsplit("_", 1)[0]
        d = out.setdefault(name, {"n": 0, "final_release": 0, "proposed_release": 0})
        d["n"] += 1
        d["final_release"] += r["final"]["action"] == "RELEASE"
        d["proposed_release"] += r["trace"].get("proposed") == "RELEASE"
    return out

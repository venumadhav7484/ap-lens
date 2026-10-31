"""Shadow mode: the pilot rehearsal.

Replays the event stream in timestamp order, as if events were arriving live. When an event
creates a decision point, the agent decides immediately through DecisionService (so every
call is audited exactly as the API would audit it). When the person's actual next action
arrives later in the stream, it is recorded against the agent's proposal. Nothing the agent
proposes changes the stream: that's what "shadow" means.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config as C
from .agent import Agent
from .service import DecisionService
from .tools import CaseStore


class RuleDecider:
    """A model client that answers with the batch rule, from the check_match tool result.
    Lets shadow mode exercise the full live path (stream → decide → audit → observe) at $0."""

    def __init__(self):
        self.messages = self

    def create(self, **kw):
        from types import SimpleNamespace as NS
        u = NS(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        results = [b for m in kw["messages"] if m["role"] == "user" and isinstance(m["content"], list)
                   for b in m["content"]]
        if not results:
            return NS(stop_reason="tool_use", usage=u, content=[NS(type="tool_use", id="m", name="check_match",
                                                                   input={})])
        m = json.loads(results[0]["content"])
        action = "RELEASE" if m["within_tolerance"] else ("WAIT_GR" if not m["gr_posted"] else "FIX_PRICE")
        ans = {"action": action, "confidence": 0.9, "reason": "batch rule", "evidence_case_ids": [],
               "buyer_note": ""}
        return NS(stop_reason="end_turn", usage=u, content=[NS(type="text", text=json.dumps(ans))])


def run(model: str = "claude-haiku-5-5", n: int = 100, client=None) -> dict:
    store = CaseStore.from_disk()
    store._nn, store._lgbm
    pts = store.points
    cutoff = pts.loc[pts["heldout"], "ts"].min()
    window = pts[(pts["ts"] >= cutoff) & pts["label_by_human"]].sort_values("ts").head(n)
    at_event = {(r[C.CASE], int(r["as_of_seq"])): r["point_id"] for _, r in window.iterrows()}
    label_at = {}
    for _, r in window.iterrows():
        rows = store.rows(r[C.CASE])
        nxt = rows.index[(rows.index > r["as_of_seq"]) & (rows[C.TS] == r["label_ts"])
                         & (rows[C.ACT] == r["label_activity"])]
        if len(nxt):
            label_at[(r[C.CASE], int(nxt[0]))] = (r["point_id"], r["label"])
    stream = store.events[store.events[C.CASE].isin(window[C.CASE])].sort_values([C.TS, C.CASE, "seq"])
    db = C.out("shadow_audit.sqlite")
    db.unlink(missing_ok=True)
    svc = DecisionService(Agent(store, model=model, client=client), db)
    decided: dict[str, int] = {}
    seq_of = {}
    t0 = time.perf_counter()
    order = store.events.sort_values([C.CASE, C.TS, "seq"])  # same ordering as decision points
    for cid, g in order[order[C.CASE].isin(window[C.CASE])].groupby(C.CASE):
        for pos, idx in enumerate(g.index):
            seq_of[idx] = pos
    for idx, ev in stream.iterrows():
        key = (ev[C.CASE], seq_of[idx])
        if key in at_event:
            decided[at_event[key]] = svc.decide(at_event[key])["id"]
        if key in label_at:
            pid, label = label_at[key]
            if pid in decided:
                svc.observe_outcome(decided[pid], label, "shadow:person")
    wall = time.perf_counter() - t0
    m = svc.metrics()
    rows = svc.export()
    agree_final = [r for r in rows if r["review_action"]]
    human_wait = window.set_index("point_id").loc[list(decided), "days_to_label"].clip(lower=0)
    out = {"model": "batch rule (no LLM)" if isinstance(client, RuleDecider) else model,
           "decision_points": len(window), "decided": len(decided),
           "outcomes_observed": len(agree_final),
           "agreement_final": (sum(r["final"] == r["review_action"] for r in agree_final) / len(agree_final))
           if agree_final else None,
           "escalation_rate": m.get("escalation_rate"), "api_errors": m.get("api_errors"),
           "agent_latency_p50_s": m.get("latency_p50_s"), "agent_latency_p95_s": m.get("latency_p95_s"),
           "person_median_days_to_act": float(np.median(human_wait)) if len(human_wait) else None,
           "cost_usd": m.get("cost_usd"), "wall_s": round(wall, 1),
           "window": [str(window["ts"].min()), str(window["ts"].max())],
           "audit_db": str(db.name)}
    C.out("shadow.json").write_text(json.dumps(out, indent=1, default=str))
    return out

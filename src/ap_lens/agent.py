"""The triage agent: a LangGraph graph with two nodes.

    investigate  - Claude reads the case through three tools and returns a typed proposal
    guardrail    - plain code: RELEASE only if the deterministic match passes, low confidence
                   becomes ESCALATE. The model cannot talk its way past this node.

Each decision writes one trace line (tool calls, tokens, cost, latency, guardrail result).
"""
from __future__ import annotations

import json
import time
from typing import Any, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from . import config as C
from .erp import ReplayERP
from .tools import CaseStore, as_json

# $ per million tokens, from the Claude API model table cached 2026-10-06. Cost figures are
# estimates from these rates; the invoice is the source of truth.
PRICES = {
    "claude-haiku-5-5": {"in": 0.10, "out": 0.50},
    "claude-sonnet-5-5": {"in": 2.00, "out": 10.00},
    "claude-opus-5-5": {"in": 4.00, "out": 20.00},
}

SYSTEM = """You triage blocked supplier invoices for an accounts-payable team.

Context: these purchase-order lines use 3-way matching with the invoice allowed before the
goods receipt. An invoice that arrives before the goods is blocked from payment. At each
decision point you decide what should happen next for the line.

Actions:
- RELEASE: goods are received and invoice, PO and receipt values agree. Payment can proceed.
- WAIT_GR: the goods receipt has not been posted yet; nothing to do until it is.
- FIX_PRICE: the invoice price disagrees with the PO; the buyer must correct the price.
- FIX_QTY: the quantities or amounts disagree in a way the buyer must correct on the PO.
- CANCEL: the invoice should be cancelled (duplicate, wrong vendor, line no longer needed).
- ESCALATE: you are not confident; a person should look.

How to work:
1. Call get_case and check_match. check_match is the authoritative 3-way match. When it reports
   a line-level "mismatch": "price" means the buyer must correct the price (FIX_PRICE);
   "quantity" means invoiced quantity exceeds what was received or ordered (FIX_QTY, or WAIT_GR
   if more goods are still expected because received is below ordered).
2. Call similar_cases to see what people did on comparable lines before.
3. Call ml_prediction (when available): a LightGBM model's probabilities, learned from earlier
   decisions. It sees vendor and item patterns the timeline doesn't show, so it is your main
   evidence for FIX_PRICE, FIX_QTY and CANCEL, which the timeline rarely reveals.
4. Decide. Prefer ESCALATE over guessing. Never propose RELEASE if check_match says
   within_tolerance is false; that will be blocked anyway.
5. For FIX_PRICE or FIX_QTY, write a short buyer_note (two sentences, plain English) the AP
   clerk could send to the buyer. Otherwise buyer_note is an empty string.

Answer with the JSON object only."""

TOOLS = [
    {"name": "get_case", "strict": True,
     "description": "Case attributes and the event timeline up to this decision point.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "check_match", "strict": True,
     "description": "Deterministic 3-way match: goods received, invoice vs PO vs receipt values within tolerance.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "ml_prediction", "strict": True,
     "description": "LightGBM class probabilities for this decision point, trained on earlier decisions.",
     "input_schema": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}},
    {"name": "similar_cases", "strict": True,
     "description": "The most similar earlier decision points and what people did next on them.",
     "input_schema": {"type": "object", "properties": {"k": {"type": "integer", "enum": [3, 5, 8]}},
                      "required": ["k"], "additionalProperties": False}},
]

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": C.AGENT_ACTIONS},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "evidence_case_ids": {"type": "array", "items": {"type": "string"}},
        "buyer_note": {"type": "string"},
    },
    "required": ["action", "confidence", "reason", "evidence_case_ids", "buyer_note"],
    "additionalProperties": False,
}


class State(TypedDict, total=False):
    point_id: str
    proposal: dict
    final: dict
    trace: dict


def cost_usd(model: str, usage: dict) -> float:
    p = PRICES.get(model)
    if not p:
        return float("nan")
    return (usage["input"] * p["in"] + usage["cache_write"] * p["in"] * 1.25
            + usage["cache_read"] * p["in"] * 0.1 + usage["output"] * p["out"]) / 1e6


class Agent:
    def __init__(self, store: CaseStore, model: str = "claude-haiku-5-5", effort: str = "low",
                 erp: ReplayERP | None = None, client: Any = None, max_turns: int = 6,
                 use_ml: bool = True, route_threshold: float | None = None):
        """route_threshold: send confident LightGBM FIX_PRICE / FIX_QTY calls straight to the
        buyer queue (router.py). None = off, the default: on the pool the rule never reached
        its 80% precision bar."""
        self.store, self.model, self.effort, self.max_turns = store, model, effort, max_turns
        self.tools = TOOLS if use_ml else [t for t in TOOLS if t["name"] != "ml_prediction"]
        self.route_threshold = route_threshold
        self.erp = erp or ReplayERP(store)
        # SDK retries 429/5xx with exponential backoff and honours retry-after; 2 retries (the
        # default) wasn't enough under a 16-worker burst on a low-tier key.
        self.client = client or anthropic.Anthropic(max_retries=8)
        g = StateGraph(State)
        g.add_node("route", self.route)
        g.add_node("investigate", self.investigate)
        g.add_node("guardrail", self.guardrail)
        g.add_edge(START, "route")
        g.add_conditional_edges("route", lambda s: "guardrail" if s.get("proposal") else "investigate",
                                ["guardrail", "investigate"])
        g.add_edge("investigate", "guardrail")
        g.add_edge("guardrail", END)
        self.graph = g.compile()

    def route(self, state: State) -> State:
        from .router import route
        if self.route_threshold is None:
            return {}
        probs = self.store.ml_prediction(state["point_id"])["probabilities"]
        hit = route(probs, self.route_threshold)
        if not hit:
            return {}
        action, p = hit
        proposal = {"action": action, "confidence": p, "evidence_case_ids": [], "buyer_note": "",
                    "reason": f"routed in code: LightGBM {action} probability {p:.2f} ≥ {self.route_threshold}"}
        trace = {"point_id": state["point_id"], "model": "router", "tool_calls": [], "stop_reason": "routed",
                 "usage": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                 "cost_usd": 0.0, "latency_s": 0.0}
        return {"proposal": proposal, "trace": trace}

    # ---- tool dispatch, bound to the decision point ----
    def _run_tool(self, point_id: str, name: str, args: dict) -> str:
        if name == "get_case":
            return as_json(self.store.get_case(point_id))
        if name == "check_match":
            return as_json(self.store.check_match(point_id))
        if name == "ml_prediction":
            return as_json(self.store.ml_prediction(point_id))
        if name == "similar_cases":
            return as_json(self.store.similar_cases(point_id, k=int(args.get("k", 5))))
        raise ValueError(name)

    def _create(self, messages: list) -> Any:
        kw = dict(model=self.model, max_tokens=4000, system=SYSTEM, tools=self.tools, messages=messages,
                  cache_control={"type": "ephemeral"},
                  output_config={"effort": self.effort,
                                 "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}})
        if self.model == "claude-sonnet-5-5":
            # Server-side refusal fallback, routed by refusal category.
            return self.client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                    fallbacks="default", **kw)
        return self.client.messages.create(**kw)

    def investigate(self, state: State) -> State:
        pid = state["point_id"]
        user = (f"Decision point {pid}. Decide what should happen next for this invoice line.")
        messages: list = [{"role": "user", "content": user}]
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        calls, t0, stop, text = [], time.perf_counter(), None, None
        for _ in range(self.max_turns):
            try:
                r = self._create(messages)
            except Exception as e:  # API down, rate-limited past retries, timeout: fail safe
                stop = f"api_error: {type(e).__name__}"
                break
            u = r.usage
            usage["input"] += u.input_tokens or 0
            usage["output"] += u.output_tokens or 0
            usage["cache_read"] += getattr(u, "cache_read_input_tokens", 0) or 0
            usage["cache_write"] += getattr(u, "cache_creation_input_tokens", 0) or 0
            stop = r.stop_reason
            if stop == "tool_use":
                messages.append({"role": "assistant", "content": r.content})
                results = []
                for b in r.content:
                    if b.type != "tool_use":
                        continue
                    calls.append(b.name)
                    try:
                        out, err = self._run_tool(pid, b.name, b.input or {}), False
                    except Exception as e:  # tool failure is reported back, not raised
                        out, err = f"error: {e}", True
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": out,
                                    "is_error": err})
                messages.append({"role": "user", "content": results})
                continue
            text = next((b.text for b in r.content if b.type == "text"), None)
            break
        proposal = _parse(text) if stop == "end_turn" else None
        if proposal is None:
            proposal = {"action": "ESCALATE", "confidence": 0.0, "evidence_case_ids": [], "buyer_note": "",
                        "reason": f"no usable answer (stop_reason={stop})"}
        trace = {"point_id": pid, "model": self.model, "effort": self.effort, "tool_calls": calls,
                 "stop_reason": stop, "usage": usage, "cost_usd": cost_usd(self.model, usage),
                 "latency_s": round(time.perf_counter() - t0, 2)}
        return {"proposal": proposal, "trace": trace}

    def guardrail(self, state: State) -> State:
        p = dict(state["proposal"])
        try:
            match = self.store.check_match(state["point_id"])
        except Exception:  # unreadable case data counts as a failed match
            match = {"within_tolerance": False}
        if p.get("action") not in C.AGENT_ACTIONS:
            p["action"] = "ESCALATE"
        try:
            conf = float(p.get("confidence", 0))
        except (TypeError, ValueError):
            conf = 0.0
        violation = p["action"] == "RELEASE" and not match["within_tolerance"]
        low_conf = p["action"] != "ESCALATE" and conf < C.MIN_CONFIDENCE
        final = dict(p)
        if violation or low_conf:
            final["action"] = "ESCALATE"
            final["guardrail"] = "release blocked: match failed" if violation else "low confidence"
        self.erp.propose(state["point_id"], final["action"],
                         {"reason": final.get("reason"), "buyer_note": final.get("buyer_note"),
                          "proposed": p["action"], "guardrail": final.get("guardrail")})
        trace = dict(state["trace"], proposed=p["action"], final=final["action"],
                     guardrail_violation=violation, low_confidence=low_conf)
        return {"final": final, "trace": trace}

    def decide(self, point_id: str) -> dict:
        out = self.graph.invoke({"point_id": point_id})
        return {"final": out["final"], "trace": out["trace"]}


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        d = json.loads(text)
    except json.JSONDecodeError:
        return None
    return d if d.get("action") in C.AGENT_ACTIONS else None

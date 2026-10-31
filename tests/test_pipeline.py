"""Offline tests on the synthetic fixture. No network, no API key: the model is a fake client."""
from __future__ import annotations

import json
import os
from types import SimpleNamespace as NS

import pytest

os.environ["APLENS_SOURCE"] = "fixture"

from ap_lens import config as C  # noqa: E402
from ap_lens import decisions, discover, fixture, guardrail_suite, load  # noqa: E402
from ap_lens.agent import Agent  # noqa: E402
from ap_lens.tools import CaseStore  # noqa: E402


@pytest.fixture(scope="session")
def store() -> CaseStore:
    fixture.write(n=1500)
    load.build()
    discover.run()
    decisions.run()
    return CaseStore.from_disk()


class FakeClient:
    """Calls each tool once, then answers with the action chosen by `decide(tool_results)`."""

    def __init__(self, decide):
        self.decide = decide
        self.messages = self

    def create(self, **kw):
        msgs = kw["messages"]
        results = [b for m in msgs if m["role"] == "user" and isinstance(m["content"], list) for b in m["content"]]
        u = NS(input_tokens=1000, output_tokens=50, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if not results:
            calls = [NS(type="tool_use", id=f"t{i}", name=n, input={"k": 5} if n == "similar_cases" else {})
                     for i, n in enumerate(["get_case", "check_match", "similar_cases", "ml_prediction"])]
            return NS(stop_reason="tool_use", content=calls, usage=u)
        tool_out = {r["tool_use_id"]: json.loads(r["content"]) for r in results}
        ans = self.decide(tool_out)
        return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(ans))], usage=u)


def test_inventory_has_expected_activities(store):
    inv = json.loads(C.out("step0_inventory.json").read_text())
    assert inv["missing_expected_activities"] == []
    assert inv["cases"] == 1500


def test_decision_points_use_only_the_prefix(store):
    p = store.points.iloc[0]
    case = store.get_case(p["point_id"])
    assert len(case["timeline"]) == p["as_of_seq"] + 1
    assert all(e["step"] <= p["as_of_seq"] for e in case["timeline"])


def test_split_has_no_leakage(store):
    pts = store.points
    cutoff = pts.loc[pts["heldout"], "ts"].min()
    assert (pts.loc[pts["pool"], "label_ts"] < cutoff).all()
    assert pts.loc[pts["heldout"], "label_by_human"].all()


def test_similar_cases_come_from_pool_only(store):
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    prec = store.similar_cases(pid)["precedents"]
    pool_ids = set(store.points.loc[store.points["pool"], "point_id"])
    assert prec and all(x["case"] in pool_ids for x in prec)


def test_guardrail_blocks_release_without_match(store):
    gstore, gids = guardrail_suite.store_with_suite(store)
    assert len(gids) == 40
    always_release = FakeClient(lambda t: {"action": "RELEASE", "confidence": 0.99, "reason": "x",
                                           "evidence_case_ids": [], "buyer_note": ""})
    agent = Agent(gstore, client=always_release)
    finals = [agent.decide(pid) for pid in gids]
    assert sum(f["final"]["action"] == "RELEASE" for f in finals) == 0
    assert all(f["trace"]["guardrail_violation"] for f in finals)


def test_guardrail_lets_a_clean_match_through(store):
    pts = store.points
    pid = pts.loc[pts["match_ok"] & pts["heldout"], "point_id"].iloc[0]
    agent = Agent(store, client=FakeClient(lambda t: {"action": "RELEASE", "confidence": 0.9, "reason": "ok",
                                                      "evidence_case_ids": [], "buyer_note": ""}))
    out = agent.decide(pid)
    assert out["final"]["action"] == "RELEASE"
    assert out["trace"]["tool_calls"] == ["get_case", "check_match", "similar_cases", "ml_prediction"]


def test_low_confidence_escalates(store):
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    agent = Agent(store, client=FakeClient(lambda t: {"action": "WAIT_GR", "confidence": 0.2, "reason": "?",
                                                      "evidence_case_ids": [], "buyer_note": ""}))
    assert agent.decide(pid)["final"]["action"] == "ESCALATE"


def test_ml_prediction_is_a_distribution(store):
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    out = store.ml_prediction(pid)
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["predicted"] in C.ACTIONS


def test_router_off_by_default_and_skips_agent_when_it_fires(store):
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    answer = {"action": "WAIT_GR", "confidence": 0.9, "reason": "x", "evidence_case_ids": [], "buyer_note": ""}
    off = Agent(store, client=FakeClient(lambda t: answer)).decide(pid)
    assert off["trace"]["model"] != "router"
    # threshold 0 always fires: the agent is never called and the action is a buyer fix
    on = Agent(store, client=FakeClient(lambda t: answer), route_threshold=0.0).decide(pid)
    assert on["trace"]["model"] == "router" and on["trace"]["tool_calls"] == []
    assert on["final"]["action"] in ("FIX_PRICE", "FIX_QTY", "ESCALATE")

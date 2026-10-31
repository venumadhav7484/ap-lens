"""Service, audit log, API and fail-safe behaviour. Offline: fake model clients only."""
from __future__ import annotations

import json
import os

import pytest

os.environ["APLENS_SOURCE"] = "fixture"

from ap_lens import config as C  # noqa: E402
from ap_lens.agent import Agent  # noqa: E402
from ap_lens.service import DecisionService  # noqa: E402
from test_pipeline import FakeClient, store  # noqa: E402,F401


class Broken:
    """A model client that fails the way real APIs do."""

    def __init__(self, mode):
        self.mode, self.messages = mode, self

    def create(self, **kw):
        from types import SimpleNamespace as NS
        u = NS(input_tokens=1, output_tokens=1, cache_read_input_tokens=0, cache_creation_input_tokens=0)
        if self.mode == "raise":
            raise ConnectionError("network down")
        if self.mode == "timeout":
            raise TimeoutError("read timed out")
        if self.mode == "max_tokens":
            return NS(stop_reason="max_tokens", content=[NS(type="text", text='{"action": "REL')], usage=u)
        if self.mode == "refusal":
            return NS(stop_reason="refusal", content=[], usage=u)
        if self.mode == "bad_json":
            return NS(stop_reason="end_turn", content=[NS(type="text", text="RELEASE it")], usage=u)
        if self.mode == "bad_action":
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(
                {"action": "PAY_NOW", "confidence": 1, "reason": "", "evidence_case_ids": [], "buyer_note": ""}))], usage=u)
        if self.mode == "bad_confidence":
            return NS(stop_reason="end_turn", content=[NS(type="text", text=json.dumps(
                {"action": "WAIT_GR", "confidence": "high", "reason": "", "evidence_case_ids": [], "buyer_note": ""}))], usage=u)
        raise AssertionError(self.mode)


@pytest.mark.parametrize("mode", ["raise", "timeout", "max_tokens", "refusal", "bad_json", "bad_action",
                                  "bad_confidence"])
def test_model_failures_fail_safe(store, mode):
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    out = Agent(store, client=Broken(mode)).decide(pid)
    assert out["final"]["action"] == "ESCALATE"


def test_audit_log_and_review_queue(store, tmp_path):
    pts = store.points[store.points["heldout"]]["point_id"].tolist()[:3]
    svc = DecisionService(Agent(store, client=FakeClient(lambda t: {
        "action": "FIX_PRICE", "confidence": 0.9, "reason": "price", "evidence_case_ids": [],
        "buyer_note": "Please correct the price."})), tmp_path / "audit.sqlite")
    ids = [svc.decide(p)["id"] for p in pts]
    assert len(svc.queue()) == 3
    svc.resolve(ids[0], "FIX_PRICE", "clerk_1")
    with pytest.raises(KeyError):
        svc.resolve(ids[0], "FIX_PRICE", "clerk_1")      # already resolved
    with pytest.raises(ValueError):
        svc.resolve(ids[1], "PAY_NOW", "clerk_1")        # not an action
    m = svc.metrics()
    assert m["decisions"] == 3 and m["reviewed"] == 1 and m["reviewer_agreement"] == 1.0
    assert len(svc.export()) == 3


def test_api_requires_key_and_round_trips(store, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from ap_lens import api
    svc = DecisionService(Agent(store, client=FakeClient(lambda t: {
        "action": "ESCALATE", "confidence": 0.3, "reason": "unclear", "evidence_case_ids": [], "buyer_note": ""})),
        tmp_path / "audit.sqlite")
    monkeypatch.setattr(api, "service", lambda: svc)
    monkeypatch.setenv("APLENS_API_KEY", "test-key")
    c = TestClient(api.app)
    pid = store.points.loc[store.points["heldout"], "point_id"].iloc[0]
    assert c.get("/healthz").status_code == 200
    assert c.post("/decisions", json={"point_id": pid}).status_code == 401
    h = {"Authorization": "Bearer test-key"}
    r = c.post("/decisions", json={"point_id": pid}, headers=h)
    assert r.status_code == 200 and r.json()["status"] == "queued"
    assert c.post("/decisions", json={"point_id": "nope#1"}, headers=h).status_code == 404
    did = c.get("/queue", headers=h).json()[0]["id"]
    assert c.post(f"/queue/{did}/resolve", json={"action": "WAIT_GR", "reviewer": "r1"}, headers=h).status_code == 200
    assert c.get("/metrics", headers=h).json()["reviewed"] == 1
    assert "Review queue" in c.get("/review", headers=h).text


def test_shadow_mode_decides_in_order_and_observes_outcomes(store):
    from ap_lens import shadow
    out = shadow.run(n=20, client=FakeClient(lambda t: {
        "action": "WAIT_GR", "confidence": 0.9, "reason": "x", "evidence_case_ids": [], "buyer_note": ""}))
    assert out["decided"] == out["decision_points"] == 20
    assert out["outcomes_observed"] > 0 and 0 <= out["agreement_final"] <= 1

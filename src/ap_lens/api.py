"""HTTP API and a minimal review page over DecisionService.

Run:  APLENS_API_KEY=... uv run uvicorn ap_lens.api:app --port 8000
Every route except /healthz needs the header  Authorization: Bearer $APLENS_API_KEY.
"""
from __future__ import annotations

import html
import os
import secrets
from functools import lru_cache

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import config as C
from .service import DecisionService


@lru_cache(maxsize=1)
def service() -> DecisionService:
    from .agent import Agent
    from .tools import CaseStore
    return DecisionService(Agent(CaseStore.from_disk(), model=os.environ.get("APLENS_MODEL", "claude-haiku-5-5")))


def require_key(authorization: str = Header(default="")) -> None:
    expected = os.environ.get("APLENS_API_KEY")
    if not expected:
        raise HTTPException(503, "APLENS_API_KEY is not set on the server")
    got = authorization.removeprefix("Bearer ").strip()
    if not secrets.compare_digest(got, expected):
        raise HTTPException(401, "bad or missing API key")


class DecideIn(BaseModel):
    point_id: str


class ResolveIn(BaseModel):
    action: str
    reviewer: str


app = FastAPI(title="AP-Lens", description="Blocked-invoice triage: proposals only, every decision audited.")


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "source": C.SOURCE}


@app.post("/decisions", dependencies=[Depends(require_key)])
def decide(body: DecideIn) -> dict:
    try:
        return service().decide(body.point_id)
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/queue", dependencies=[Depends(require_key)])
def queue(limit: int = 50) -> list[dict]:
    return service().queue(limit)


@app.post("/queue/{decision_id}/resolve", dependencies=[Depends(require_key)])
def resolve(decision_id: int, body: ResolveIn) -> dict:
    try:
        return service().resolve(decision_id, body.action, body.reviewer)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except KeyError as e:
        raise HTTPException(404, str(e))


@app.get("/metrics", dependencies=[Depends(require_key)])
def metrics() -> dict:
    return service().metrics()


@app.get("/review", response_class=HTMLResponse, dependencies=[Depends(require_key)])
def review_page() -> str:
    rows = service().queue(100)
    body = "".join(
        f"<tr><td>{r['id']}</td><td>{html.escape(r['point_id'])}</td><td><b>{r['final']}</b>"
        f"{' (' + html.escape(r['guardrail']) + ')' if r['guardrail'] else ''}</td>"
        f"<td>{html.escape(r['reason'] or '')}</td><td>{html.escape(r['buyer_note'] or '')}</td></tr>"
        for r in rows) or "<tr><td colspan=5>Queue is empty.</td></tr>"
    return ("<!doctype html><meta charset=utf-8><title>AP-Lens review queue</title>"
            "<style>body{font:14px system-ui;margin:24px}td,th{border-bottom:1px solid #ddd;padding:6px;text-align:left}</style>"
            "<h1>Review queue</h1><p>Resolve with POST /queue/{id}/resolve.</p>"
            "<table><tr><th>id</th><th>decision point</th><th>proposal</th><th>reason</th><th>buyer note</th></tr>"
            + body + "</table>")

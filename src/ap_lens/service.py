"""DecisionService: the pilot-shaped wrapper around the agent.

Every decision lands in an append-only SQLite audit log. Nothing is ever executed
automatically: RELEASE and WAIT_GR are recorded as proposals; FIX_*, CANCEL and ESCALATE go
to a human review queue. A reviewer's action is recorded next to the agent's, which gives
the live agreement number a pilot would be judged on.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as C

REVIEW_ACTIONS = {"FIX_PRICE", "FIX_QTY", "CANCEL", "ESCALATE"}

SCHEMA = """
create table if not exists decisions (
  id integer primary key autoincrement,
  created_at text not null,
  point_id text not null,
  model text, proposed text, final text not null, confidence real,
  guardrail text, reason text, buyer_note text,
  cost_usd real, latency_s real, stop_reason text,
  status text not null,              -- proposed | queued | resolved
  reviewer text, review_action text, reviewed_at text
);
create index if not exists decisions_status on decisions(status);
create index if not exists decisions_point on decisions(point_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionService:
    def __init__(self, agent, db_path: Path | str | None = None):
        self.agent = agent
        self.db_path = str(db_path or C.out("audit.sqlite"))
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False)
        c.row_factory = sqlite3.Row
        return c

    def decide(self, point_id: str) -> dict:
        self.agent.store.point(point_id)  # unknown id -> KeyError, never an audited ESCALATE
        out = self.agent.decide(point_id)
        f, t = out["final"], out["trace"]
        status = "queued" if f["action"] in REVIEW_ACTIONS else "proposed"
        row = (now(), point_id, t.get("model"), t.get("proposed"), f["action"], _num(f.get("confidence")),
               f.get("guardrail"), f.get("reason"), f.get("buyer_note"), _num(t.get("cost_usd")),
               _num(t.get("latency_s")), t.get("stop_reason"), status)
        with self._lock, self._conn() as c:
            cur = c.execute("""insert into decisions (created_at, point_id, model, proposed, final, confidence,
                guardrail, reason, buyer_note, cost_usd, latency_s, stop_reason, status)
                values (?,?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            did = cur.lastrowid
        return {"id": did, "status": status, **{k: f.get(k) for k in ("action", "confidence", "reason",
                                                                       "buyer_note", "guardrail")}}

    def queue(self, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("select * from decisions where status = 'queued' order by id limit ?", (limit,))
            return [dict(r) for r in rows]

    def resolve(self, decision_id: int, action: str, reviewer: str) -> dict:
        if action not in C.ACTIONS:
            raise ValueError(f"action must be one of {C.ACTIONS}")
        with self._lock, self._conn() as c:
            n = c.execute("""update decisions set status = 'resolved', review_action = ?, reviewer = ?,
                             reviewed_at = ? where id = ? and status = 'queued'""",
                          (action, reviewer, now(), decision_id)).rowcount
        if not n:
            raise KeyError(f"decision {decision_id} is not in the queue")
        return {"id": decision_id, "status": "resolved", "review_action": action}

    def observe_outcome(self, decision_id: int, action: str, observer: str = "observed") -> None:
        """Shadow mode: record what actually happened next, whatever the decision's status."""
        with self._lock, self._conn() as c:
            c.execute("""update decisions set status = 'resolved', review_action = ?, reviewer = ?,
                         reviewed_at = ? where id = ?""", (action, observer, now(), decision_id))

    def metrics(self) -> dict:
        with self._conn() as c:
            rows = [dict(r) for r in c.execute("select * from decisions")]
        if not rows:
            return {"decisions": 0}
        lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
        reviewed = [r for r in rows if r["status"] == "resolved"]
        by = lambda k: {v: sum(1 for r in rows if r[k] == v) for v in sorted({r[k] for r in rows if r[k]})}
        return {
            "decisions": len(rows),
            "by_final_action": by("final"),
            "by_status": by("status"),
            "escalation_rate": sum(r["final"] == "ESCALATE" for r in rows) / len(rows),
            "guardrail_blocks": sum(1 for r in rows if r["guardrail"]),
            "api_errors": sum(1 for r in rows if (r["stop_reason"] or "").startswith("api_error")),
            "cost_usd": float(sum(r["cost_usd"] or 0 for r in rows)),
            "latency_p50_s": float(np.percentile(lat, 50)) if lat else None,
            "latency_p95_s": float(np.percentile(lat, 95)) if lat else None,
            "reviewed": len(reviewed),
            "reviewer_agreement": (sum(r["review_action"] == r["proposed"] for r in reviewed) / len(reviewed))
            if reviewed else None,
        }

    def export(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("select * from decisions order by id")]


def _num(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def dumps(x) -> str:
    return json.dumps(x, default=str)

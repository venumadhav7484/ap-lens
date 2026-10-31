"""The integration seam. A real deployment implements ERPAdapter against SAP (or any ERP);
AP-Lens ships ReplayERP, which reads the public log and records proposals as a dry run.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Protocol

from . import config as C
from .tools import CaseStore


class ERPAdapter(Protocol):
    def read_case(self, point_id: str) -> dict: ...
    def propose(self, point_id: str, action: str, payload: dict) -> dict: ...


class ReplayERP:
    """Never mutates anything. Every proposal is appended to outputs/proposals.jsonl."""

    def __init__(self, store: CaseStore, log_name: str = "proposals.jsonl"):
        self.store = store
        self.path = C.out(log_name)

    def read_case(self, point_id: str) -> dict:
        return self.store.get_case(point_id)

    def propose(self, point_id: str, action: str, payload: dict) -> dict:
        if action not in C.AGENT_ACTIONS:
            raise ValueError(f"unknown action {action}")
        rec = {"at": datetime.now(timezone.utc).isoformat(), "point_id": point_id,
               "action": action, "dry_run": True, **payload}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return {"recorded": True, "dry_run": True}

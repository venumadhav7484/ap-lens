"""AP-Lens tools over MCP, so Claude Code, Codex or Cursor can investigate a blocked invoice.

Run:  uv run ap-lens-mcp            (stdio)
Add to Claude Code:  claude mcp add ap-lens -- uv --directory <repo> run ap-lens-mcp
"""
from __future__ import annotations

from functools import lru_cache

from mcp.server.mcpserver import MCPServer

from . import config as C
from .erp import ReplayERP
from .tools import CaseStore

server = MCPServer(
    name="ap-lens",
    instructions="Read-only tools over a replayed purchase-to-pay log. Every read stops at the "
                 "decision point. propose_action records a dry-run proposal; it never changes data.",
)


@lru_cache(maxsize=1)
def store() -> CaseStore:
    return CaseStore.from_disk()


@server.tool(description="List decision points from the held-out set (blocked invoices awaiting a decision).")
def list_decision_points(limit: int = 20) -> list[dict]:
    p = store().points
    rows = p[p["heldout"]].head(limit)
    return [{"point_id": r.point_id, "triggered_by": r.point_type, "item_type": r.item_type,
             "spend_area": r.spend_area_text} for r in rows.itertuples()]


@server.tool(description="Case attributes and the event timeline up to the decision point.")
def get_case(point_id: str) -> dict:
    return store().get_case(point_id)


@server.tool(description="Deterministic 3-way match at the decision point.")
def check_match(point_id: str) -> dict:
    return store().check_match(point_id)


@server.tool(description="Most similar earlier decision points and what people did next.")
def similar_cases(point_id: str, k: int = 5) -> dict:
    return store().similar_cases(point_id, k=k)


@server.tool(description="LightGBM class probabilities for the decision point, trained on earlier decisions.")
def ml_prediction(point_id: str) -> dict:
    return store().ml_prediction(point_id)


@server.tool(description="Record a proposed action as a dry run. RELEASE is refused unless the match passes.")
def propose_action(point_id: str, action: str, reason: str, buyer_note: str = "") -> dict:
    if action == "RELEASE" and not store().check_match(point_id)["within_tolerance"]:
        return {"recorded": False, "refused": "RELEASE needs a passing 3-way match"}
    return ReplayERP(store(), "mcp_proposals.jsonl").propose(
        point_id, action, {"reason": reason, "buyer_note": buyer_note, "via": "mcp"})


@server.resource("policy://release-rules")
def policy() -> str:
    return (f"RELEASE only when goods are received and invoice, PO and receipt values agree within "
            f"{C.VALUE_TOLERANCE:.0%}. Confidence below {C.MIN_CONFIDENCE} becomes ESCALATE.")


def main() -> None:
    server.run("stdio")

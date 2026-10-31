"""Step 1: discovery. Where does work wait, who does it, and which bucket does each step belong in.

All numbers come from SQL over the event log and land in outputs/discovery.json.
"""
from __future__ import annotations

import json

import duckdb

from . import config as C

# delete / code / agent / human. Judgement calls, written down so they can be argued with.
BUCKET_OF = {
    C.CLEAR: ("code", "Payment run; fully decided by fields"),
    "Vendor creates invoice": ("code", "Vendor-side event; capture via e-invoice"),
    C.INV: ("code", "Invoice capture: e-invoicing / OCR, not a judgement"),
    C.UNBLOCK: ("agent", "Release decision: code checks the match, agent triages the rest"),
    C.CHG_PRICE: ("agent", "Agent drafts the buyer note; buyer approves"),
    C.CHG_QTY: ("agent", "Agent drafts the buyer note; buyer approves"),
    C.CREATE: ("human", "Buying decision"),
    C.GR: ("human", "Physical receipt at the warehouse"),
    C.CANCEL_INV: ("human", "Money call"),
    C.DEL_ITEM: ("human", "Money call"),
    C.CANCEL_GR: ("human", "Warehouse correction"),
    "Change Approval for Purchase Order": ("delete", "Repeated approval changes rarely change the outcome"),
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"create view events as select * from read_parquet('{C.parquet('events')}')")
    con.execute(f"create view cases as select * from read_parquet('{C.parquet('cases')}')")
    con.create_function("is_human", lambda r: C.is_human(r), ["VARCHAR"], "BOOLEAN")
    return con


def q(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def run() -> dict:
    con = connect()
    d: dict = {"source": C.SOURCE}
    d["overview"] = q(con, f"""
        select count(*) events, count(distinct {C.CASE}) cases, count(distinct {C.ACT}) activities,
               avg(case when is_human({C.RES}) then 1 else 0 end) human_share
        from events""")[0]
    d["flows"] = q(con, f"""
        select {C.ITEM_CATEGORY} flow, count(*) cases, count(*) / sum(count(*)) over () share
        from cases group by 1 order by 2 desc""")
    d["activities"] = q(con, f"""
        with e as (select *, lead({C.TS}) over (partition by {C.CASE} order by {C.TS}, seq) nxt from events),
        per as (select {C.CASE}, {C.ACT}, count(*) n from events group by 1, 2)
        select e.{C.ACT} activity, count(*) events, count(distinct e.{C.CASE}) cases,
               avg(case when is_human(e.{C.RES}) then 1 else 0 end) human_share,
               median(date_diff('second', e.{C.TS}, e.nxt) / 86400.0) median_days_to_next,
               (select avg(case when n > 1 then 1 else 0 end) from per p where p.{C.ACT} = e.{C.ACT}) repeat_share
        from e group by 1 order by 2 desc""")
    d["variants"] = q(con, f"""
        with v as (select {C.CASE}, string_agg({C.ACT}, ' > ' order by {C.TS}, seq) path from events group by 1)
        select path, count(*) cases, count(*) / sum(count(*)) over () share from v group by 1 order by 2 desc limit 10""")
    d["n_variants"] = q(con, f"""
        select count(distinct path) n from (select string_agg({C.ACT}, '>' order by {C.TS}, seq) path
        from events group by {C.CASE})""")[0]["n"]
    d["slowest_handoffs"] = q(con, f"""
        with e as (select {C.ACT} a, lead({C.ACT}) over w b, date_diff('second', {C.TS}, lead({C.TS}) over w) / 86400.0 as gap_days
                   from events window w as (partition by {C.CASE} order by {C.TS}, seq))
        select a || ' → ' || b handoff, count(*) n, median(gap_days) median_days from e
        where b is not null group by 1 having count(*) >= 50 order by median_days desc limit 10""")
    d["block_flow"] = q(con, f"""
        with c as (select {C.CASE} from cases where {C.ITEM_CATEGORY} = '{C.BLOCK_FLOW}'),
        e as (select e.* from events e join c using ({C.CASE})),
        t as (select {C.CASE},
                     min(case when {C.ACT} = '{C.INV}' then {C.TS} end) inv,
                     max(case when {C.ACT} = '{C.CLEAR}' then {C.TS} end) clr from e group by 1)
        select (select count(*) from c) cases,
               (select count(*) from e where {C.ACT} = '{C.UNBLOCK}') unblocks,
               (select avg(case when is_human({C.RES}) then 1 else 0 end) from e where {C.ACT} = '{C.UNBLOCK}') unblock_human_share,
               (select median(date_diff('second', inv, clr) / 86400.0) from t where inv is not null and clr > inv) median_days_inv_to_clear""")[0]
    buckets: dict[str, dict] = {}
    for a in d["activities"]:
        b, why = BUCKET_OF.get(a["activity"], ("unassigned", "Review: not mapped yet"))
        s = buckets.setdefault(b, {"events": 0, "human_events": 0, "activities": []})
        s["events"] += a["events"]
        s["human_events"] += round(a["events"] * (a["human_share"] or 0))
        s["activities"].append({"activity": a["activity"], "events": a["events"], "why": why})
    d["buckets"] = buckets
    C.out("discovery.json").write_text(json.dumps(d, indent=1, default=str))
    return d

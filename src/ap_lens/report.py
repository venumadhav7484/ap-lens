"""Render outputs/*.json into one HTML page and the README numbers block.
No number is typed by hand: everything comes from the JSON written by earlier steps.
"""
from __future__ import annotations

import html
import json
import re

from . import config as C


def _j(name: str):
    p = C.out(name)
    return json.loads(p.read_text()) if p.exists() else None


def _pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def best_baseline(m: dict) -> tuple[str, dict]:
    return max(m["baselines"].items(), key=lambda kv: kv[1]["macro_f1"])


def headline(model: str) -> dict | None:
    m, k = _j("metrics.json"), _j(f"kpi_{model}.json")
    if not m or model not in m.get("agents", {}):
        return None
    a = m["agents"][model]
    fname, fb = best_baseline(m)
    aname, ab = max(m["baselines"].items(), key=lambda kv: kv[1]["accuracy"])
    srp = {n: v["safe_release_precision"] or 0 for n, v in m["baselines"].items()}
    sname = max(srp, key=srp.get)
    labels = a["confusion"]["labels"]
    mx = a["confusion"]["matrix"]
    minority = [l for l in ("FIX_PRICE", "FIX_QTY", "CANCEL") if l in labels]
    caught = sum(mx[labels.index(l)][labels.index(l)] for l in minority)
    total_minority = sum(sum(mx[labels.index(l)]) for l in minority)
    return {"model": model, "n": a["completed"], "agent_acc": a["accuracy"], "agent_f1": a["macro_f1"],
            "agent_srp": a["safe_release_precision"], "abstain": a["abstain_rate"],
            "acc_base": aname, "acc_base_acc": ab["accuracy"], "f1_base": fname, "f1_base_f1": fb["macro_f1"],
            "srp_base": sname, "srp_base_srp": srp[sname],
            "minority_caught": caught, "minority_total": total_minority,
            "unsafe": a["guardrail_suite"]["final_releases"], "suite": a["guardrail_suite"]["cases"],
            "cost_1k": a["cost_usd_per_1000"], "p50": a["latency_p50_s"],
            "touches_after": k and k["manual_touches_per_100_after"],
            "wrong_releases": k and k["wrong_releases"], "source": m["source"]}


def readme_block(h: dict) -> str:
    tag = "" if h["source"] == "real" else " (SYNTHETIC FIXTURE: placeholder, not a result)"
    data = C.XES_SOURCES[C.XES_SOURCE]["note"] if h["source"] == "real" else "synthetic fixture"
    return "\n".join([
        f"**{h['n']} held-out human decisions{tag}** · data: {data}",
        "",
        f"| | Agent ({h['model']}) | Best non-LLM baseline |",
        "|---|---|---|",
        f"| Agreement with the person | **{_pct(h['agent_acc'])}** | {_pct(h['acc_base_acc'])} ({h['acc_base']}) |",
        f"| Macro-F1 (all 5 actions) | **{h['agent_f1']:.2f}** | {h['f1_base_f1']:.2f} ({h['f1_base']}) |",
        f"| Safe-release precision | **{_pct(h['agent_srp'])}** | {_pct(h['srp_base_srp'])} ({h['srp_base']}) |",
        f"| Price / quantity / cancel cases caught | {h['minority_caught']} of {h['minority_total']} | see report |",
        f"| Unsafe releases, {h['suite']}-case guardrail suite | **{h['unsafe']}** | — |",
        "",
        f"Replay (simulated): manual touches fall to **{h['touches_after']:.0f} per 100**, with "
        f"**{h['wrong_releases']} releases where the person did something else**. "
        f"Agent escalates {_pct(h['abstain'])}. Cost ≈ ${h['cost_1k']:.2f} per 1,000 decisions, "
        f"p50 {h['p50']:.1f}s.",
    ])


def update_readme(model: str) -> None:
    h = headline(model)
    if not h:
        return
    p = C.ROOT / "README.md"
    txt = p.read_text()
    new = f"<!-- numbers:start -->\n{readme_block(h)}\n<!-- numbers:end -->"
    txt = re.sub(r"<!-- numbers:start -->.*?<!-- numbers:end -->", new, txt, flags=re.S)
    p.write_text(txt)


def _detail(name: str, r: dict) -> str:
    if name == "fuzz":
        return (f"{r['cases']:,} random cases, {r['model_proposed_release']:,} release proposals, "
                f"{r['releases_without_match']} released without a match")
    if name == "failures":
        m = r["modes"]
        return (f"{len(m)} failure types × {next(iter(m.values()))['decisions']}: "
                f"{sum(v['escalated'] for v in m.values())} escalated, {sum(v['crashes'] for v in m.values())} crashes")
    if name == "drift":
        return ", ".join(f"{k}: {v.get('crashes', 'not built')} crashes" for k, v in r["mutations"].items())
    if name == "injection":
        return (f"model proposed RELEASE {r['control']['model_proposed_release']}/{r['control']['n']} clean vs "
                f"{r['injected']['model_proposed_release']}/{r['injected']['n']} injected; final releases "
                f"{r['injected']['final_release']}")
    if name == "amounts":
        b = r["invoice_plus_15pct"]
        return (f"invoice +15%: RELEASE {b.get('RELEASE', 0)}, FIX_PRICE {b.get('FIX_PRICE', 0)}, "
                f"FIX_QTY {b.get('FIX_QTY', 0)}, ESCALATE {b.get('ESCALATE', 0)} of {b['n']}")
    if name == "load":
        return (f"{r['decisions']:,} decisions, {r['decisions_per_minute']}/min with {r['workers']} workers, "
                f"{r['api_errors']} API errors, p95 {r['latency_p95_s']:.1f}s, ${r['cost_per_1000']:.2f}/1,000")
    return ""


def _table(rows, cols):
    head = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(r.get(c, '')))}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(model: str) -> str:
    d, m, k, s = _j("discovery.json"), _j("metrics.json"), _j(f"kpi_{model}.json"), _j("decisions_summary.json")
    warn = "" if C.SOURCE == "real" else "<p class='warn'>SYNTHETIC FIXTURE — pipeline check only. Not results.</p>"
    parts = [f"<h1>AP-Lens results</h1>{warn}"]
    if d:
        bf = d["block_flow"]
        parts.append("<h2>Discovery</h2>")
        parts.append(f"<p>{d['overview']['events']:,} events, {d['overview']['cases']:,} cases, "
                     f"{d['n_variants']:,} distinct paths. Blocked-invoice flow: {bf['cases']:,} cases, "
                     f"{bf['unblocks']:,} unblocks, {_pct(bf['unblock_human_share'])} by people; median "
                     f"{bf['median_days_inv_to_clear']:.1f} days from invoice to clearing.</p>")
        parts.append(_table([{"bucket": b, "events": v["events"], "by people": v["human_events"],
                              "activities": ", ".join(a["activity"] for a in v["activities"])}
                             for b, v in d["buckets"].items()], ["bucket", "events", "by people", "activities"]))
        parts.append(_table([{k2: (round(v, 1) if isinstance(v, float) else v) for k2, v in r.items()}
                             for r in d["slowest_handoffs"]], ["handoff", "n", "median_days"]))
    if s:
        parts.append(f"<h2>Decision points</h2><p>{s['decision_points']:,} decision points, "
                     f"{s['human_decisions']:,} decided by people (gate ≥ 300: {'pass' if s['gate_ok'] else 'FAIL'}). "
                     f"Held-out: {s['heldout']} after {s['cutoff'][:10]}.</p>")
    if m:
        rows = [{"system": n, "accuracy": _pct(v["accuracy"]), "macro-F1": f"{v['macro_f1']:.2f}",
                 "abstain": _pct(v["abstain_rate"]), "safe-release precision": _pct(v["safe_release_precision"]),
                 "releases without match": v["releases_without_match"]}
                for n, v in [*m["baselines"].items(), *m["agents"].items()]]
        parts.append("<h2>Eval</h2>" + _table(rows, list(rows[0])))
        for name, a in m["agents"].items():
            g = a["guardrail_suite"]
            parts.append(f"<p><b>{name}</b>: guardrail suite {g['final_releases']} unsafe releases of {g['cases']} "
                         f"(model proposed {g['model_proposed_releases']}, blocked by code). "
                         f"${a['cost_usd_total']:.2f} total, ${a['cost_usd_per_1000']:.2f} per 1,000; "
                         f"p50 {a['latency_p50_s']:.1f}s, p95 {a['latency_p95_s']:.1f}s; "
                         f"{a['mean_tool_calls']:.1f} tool calls per decision.</p>")
    if k:
        parts.append("<h2>KPI (simulated)</h2>" + _table([
            {"KPI": "Manual touches per 100 decisions", "before": 100, "after": round(k["manual_touches_per_100_after"], 1)},
            {"KPI": "Median days, block → release (released lines)",
             "before": round(k["median_days_block_to_release_before"] or 0, 1),
             "after": round(k["median_days_block_to_release_after"] or 0, 1)},
            {"KPI": "Releases the agent got wrong", "before": "—", "after": k["wrong_releases"]},
        ], ["KPI", "before", "after"]) + f"<p class='note'>{html.escape(k['note'])}</p>")
    rd = _j("readiness.json")
    if rd:
        parts.append("<h2>Readiness scorecard</h2>" + _table(
            [{"check": n, "result": {True: "PASS", False: "FAIL"}.get(rd[n]["pass"], "INCONCLUSIVE"), "detail": html.escape(_detail(n, rd[n]))}
             for n in ("fuzz", "failures", "drift", "injection", "amounts", "load") if n in rd],
            ["check", "result", "detail"]))
    sh = _j("shadow.json")
    if sh:
        parts.append("<h2>Shadow mode</h2>" + _table([{k2: v for k2, v in sh.items() if k2 != "window"}],
                                                       [k2 for k2 in sh if k2 != "window"]))
    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;color:#1c2230}"
           "table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border-bottom:1px solid #ddd;padding:6px 8px;"
           "text-align:left;font-size:13px}.warn{background:#fde8d8;padding:8px 12px;border-radius:4px}"
           ".note{color:#666;font-size:13px}")
    page = f"<!doctype html><meta charset='utf-8'><title>AP-Lens results</title><style>{css}</style>" + "".join(parts)
    C.out("report.html").write_text(page)
    return page

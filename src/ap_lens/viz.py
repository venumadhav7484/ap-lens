"""README figures, generated from outputs/*.json (no number typed by hand).

Static SVGs that follow GitHub's light/dark theme via prefers-color-scheme. Palette: the
dataviz reference categorical slots 1-4 (validated light and dark; light-mode aqua/yellow sit
below 3:1 on the surface, so every bar carries a direct value label and the README keeps the
table view). Color follows the entity: Agent = slot 1, batch rule = slot 2, LightGBM = slot 3.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from . import config as C

IMG = C.ROOT / "docs" / "img"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
STYLE = f"""<style>
.bg{{fill:#fcfcfb;stroke:#e3e2dd}} .t1{{fill:#0b0b0b}} .t2{{fill:#52514e}} .grid{{stroke:#e3e2dd}} .base{{stroke:#8a8984}}
.s1{{fill:#2a78d6}} .s2{{fill:#eb6834}} .s3{{fill:#1baf7a}} .s4{{fill:#eda100}}
.l1{{stroke:#2a78d6}} .ref{{stroke:#52514e}}
text{{font-family:{FONT}}}
@media (prefers-color-scheme: dark){{
.bg{{fill:#1a1a19;stroke:#3a3936}} .t1{{fill:#f0efec}} .t2{{fill:#c3c2b7}} .grid{{stroke:#3a3936}} .base{{stroke:#6f6e69}}
.s1{{fill:#3987e5}} .s2{{fill:#d95926}} .s3{{fill:#199e70}} .s4{{fill:#c98500}}
.l1{{stroke:#3987e5}} .ref{{stroke:#c3c2b7}}
}}
</style>"""

SYSTEMS = [("Agent (Haiku 5.5)", "s1"), ("Batch rule", "s2"), ("LightGBM", "s3")]


def _j(name: str):
    p = C.out(name)
    return json.loads(p.read_text()) if p.exists() else None


def _svg(w: int, h: int, title: str, desc: str, body: str) -> str:
    w, h = w + 28, h + 20
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-labelledby="t d"><title id="t">{html.escape(title)}</title>'
            f'<desc id="d">{html.escape(desc)}</desc>{STYLE}'
            f'<rect class="bg" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="8"/>'
            f'<g transform="translate(14,10)">{body}</g></svg>')


def _bar(x: float, y: float, w: float, h: float, cls: str) -> str:
    """Horizontal bar anchored at x, 4px rounded data-end only."""
    if w <= 0:
        return ""
    r = min(4.0, w / 2, h / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def _text(x, y, s, cls="t1", size=12, anchor="start", weight=400) -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}">{html.escape(str(s))}</text>')


def _legend(x: float, y: float, items: list[tuple[str, str]]) -> str:
    out, cx = [], x
    for label, cls in items:
        out.append(f'<rect x="{cx:.1f}" y="{y - 9:.1f}" width="10" height="10" rx="2" class="{cls}"/>')
        out.append(_text(cx + 15, y, label, "t2", 12))
        cx += 22 + 7 * len(label)
    return "".join(out)


def _systems(m: dict, model: str) -> list[tuple[str, str, dict]]:
    a = m["agents"][model]
    return [("Agent (Haiku 5.5)", "s1", a), ("Batch rule", "s2", m["baselines"]["batch_rule"]),
            ("LightGBM", "s3", m["baselines"]["lightgbm"])]


def fig_compare(m: dict, model: str) -> str:
    rows = _systems(m, model)
    panels = [("Agreement with the person", "accuracy", True),
              ("Macro-F1 (all 5 actions)", "macro_f1", False),
              ("Safe-release precision", "safe_release_precision", True)]
    W, pw, gap, left, top, bh = 760, 200, 30, 10, 48, 16
    body = [_text(left, 18, f"Agent vs non-LLM baselines on {rows[0][2]['n']} held-out human decisions",
                  "t1", 14, weight=600), _legend(left, 36, [(n, c) for n, c, _ in rows])]
    for i, (title, key, pct) in enumerate(panels):
        x0 = left + i * (pw + gap)
        body.append(_text(x0, top + 6, title, "t2", 12, weight=600))
        y = top + 18
        body.append(f'<line class="base" x1="{x0}" y1="{y - 2}" x2="{x0}" y2="{y + 3 * (bh + 2) + 2}"/>')
        for name, cls, v in rows:
            val = v.get(key) or 0.0
            w = (pw - 46) * val
            body.append(_bar(x0, y, w, bh, cls))
            body.append(_text(x0 + w + 5, y + 12, f"{100 * val:.1f}%" if pct else f"{val:.2f}", "t1", 12))
            y += bh + 2
    return _svg(W, top + 18 + 3 * (bh + 2) + 12, "Agent vs baselines",
                "; ".join(f"{n}: agreement {v['accuracy']:.3f}, macro-F1 {v['macro_f1']:.2f}, safe-release "
                          f"precision {v.get('safe_release_precision') or 0:.3f}" for n, _, v in rows), "".join(body))


def fig_classes(m: dict, model: str) -> str:
    rows = _systems(m, model)
    labels = ["RELEASE", "WAIT_GR", "FIX_PRICE", "FIX_QTY", "CANCEL"]
    W, left, label_w, top, bh, plot_w = 760, 10, 100, 46, 11, 520
    body = [_text(left, 18, "Who catches what: share of each action the system got right", "t1", 14, weight=600),
            _legend(left, 36, [(n, c) for n, c, _ in rows])]
    y = top + 6
    for lab in labels:
        body.append(_text(left, y + 20, lab, "t1", 12, weight=600))
        x0 = left + label_w
        body.append(f'<line class="base" x1="{x0}" y1="{y - 2}" x2="{x0}" y2="{y + 3 * (bh + 2)}"/>')
        for name, cls, v in rows:
            L, M = v["confusion"]["labels"], v["confusion"]["matrix"]
            i = L.index(lab)
            hit, tot = M[i][i], sum(M[i])
            frac = hit / tot if tot else 0
            w = plot_w * frac
            body.append(_bar(x0, y, w, bh, cls))
            body.append(_text(x0 + w + 5, y + 10, f"{hit}/{tot}", "t2", 11))
            y += bh + 2
        y += 12
    return _svg(W, y + 4, "Per-action recall", "Correct over total, per action, for agent, batch rule and "
                "LightGBM", "".join(body))


def fig_amounts(rd: dict) -> str:
    a = rd["amounts"]
    segs = [("RELEASE", "s1"), ("FIX_PRICE", "s2"), ("ESCALATE", "s3"), ("CANCEL", "s4")]
    W, left, label_w, top, bh, plot_w = 760, 10, 170, 50, 26, 510
    body = [_text(left, 18, "Give the agent invoice amounts and it stops releasing mismatches", "t1", 14, weight=600),
            _legend(left, 38, [(s, c) for s, c in segs])]
    y = top
    for key, label in (("control", "Real cases, as logged"), ("invoice_plus_15pct", "Same cases, invoice +15%")):
        d = a[key]
        n = d["n"]
        body.append(_text(left, y + 17, label, "t1", 12))
        x = left + label_w
        for s, cls in segs:
            v = d.get(s, 0)
            if not v:
                continue
            w = plot_w * v / n
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w - 2, 0):.1f}" height="{bh}" rx="3" class="{cls}"/>')
            if w > 26:
                body.append(_text(x + (w - 2) / 2, y + 17, v, "t1", 12, "middle", 600))
            else:  # too narrow to hold its number: label it just above
                body.append(_text(x + (w - 2) / 2, y - 3, v, "t1", 11, "middle", 600))
            x += w
        body.append(_text(x + 4, y + 17, f"of {n}", "t2", 11))
        y += bh + 14
    return _svg(W, y + 4, "Invoice-amount counterfactual",
                f"Control: {a['control']}. Invoice +15%: {a['invoice_plus_15pct']}", "".join(body))


def fig_router(r: dict) -> str:
    sweep = [s for s in r["sweep"] if s["precision"] is not None]
    target = r["target_precision"]
    W, H, left, top, pw, ph = 760, 250, 56, 40, 640, 160
    xs = [s["threshold"] for s in sweep]
    x_min, x_max = min(xs), max(xs)
    X = lambda t: left + pw * (t - x_min) / (x_max - x_min)
    Y = lambda p: top + ph * (1 - p)
    body = [_text(10, 18, "Routing LightGBM's price/quantity calls: precision never reaches the 80% bar",
                  "t1", 14, weight=600)]
    for p in (0, 0.25, 0.5, 0.75, 1.0):
        body.append(f'<line class="grid" x1="{left}" y1="{Y(p):.1f}" x2="{left + pw}" y2="{Y(p):.1f}"/>')
        body.append(_text(left - 8, Y(p) + 4, f"{int(p * 100)}%", "t2", 11, "end"))
    body.append(f'<line class="ref" stroke-dasharray="5 4" stroke-width="1.5" x1="{left}" y1="{Y(target):.1f}" '
                f'x2="{left + pw}" y2="{Y(target):.1f}"/>')
    body.append(_text(left + 6, Y(target) - 6, f"bar set beforehand: {int(target * 100)}%", "t2", 11))
    pts = " ".join(f"{X(s['threshold']):.1f},{Y(s['precision']):.1f}" for s in sweep)
    body.append(f'<polyline class="l1" fill="none" stroke-width="2" points="{pts}"/>')
    for s in sweep:
        body.append(f'<circle class="s1" cx="{X(s["threshold"]):.1f}" cy="{Y(s["precision"]):.1f}" r="4" '
                    f'stroke="none"/>')
    last = sweep[-1]
    body.append(_text(X(last["threshold"]), Y(0.9) + 4,
                      f"{100 * last['precision']:.0f}% at {last['threshold']:.2f} ({last['routed']} lines)",
                      "t1", 12, "end", 600))
    for t in xs[::2]:
        body.append(_text(X(t), top + ph + 16, f"{t:.2f}", "t2", 11, "middle"))
    body.append(_text(left + pw / 2, top + ph + 34, "LightGBM confidence threshold (5-fold CV on the precedent pool)",
                      "t2", 11, "middle"))
    return _svg(W, H, "Router precision sweep", "; ".join(f"{s['threshold']:.2f}: {s['precision']:.2f}"
                                                         for s in sweep), "".join(body))


def write_all(model: str = "claude-haiku-5-5") -> list[Path]:
    IMG.mkdir(parents=True, exist_ok=True)
    out = []
    m, rd, r = _j("metrics.json"), _j("readiness.json"), _j("router_threshold.json")
    figs = []
    if m and model in m.get("agents", {}):
        figs += [("compare.svg", fig_compare(m, model)), ("per_action.svg", fig_classes(m, model))]
    if rd and "amounts" in rd:
        figs.append(("amounts.svg", fig_amounts(rd)))
    if r:
        figs.append(("router.svg", fig_router(r)))
    for name, svg in figs:
        p = IMG / name
        p.write_text(svg)
        out.append(p)
    return out

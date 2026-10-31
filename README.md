# AP-Lens

**A forward-deployed engineering case study, replayed on public data.** AP-Lens finds where accounts-payable work stalls in a real purchase-to-pay log, hands one exception (blocked invoices) to a tool-using agent behind code-level guardrails, and scores it against what people actually did, next to three non-LLM baselines.

> **Replayed, not deployed.** Everything runs on a public historical log (BPI Challenge 2019). No client, no live ERP. Every "after" figure is simulated.

## Results

<!-- numbers:start -->
**500 held-out human decisions** · data: Subset of BPIC19 (Zenodo 17295283, CC BY 4.0): 217,406 of 251,734 cases.

| | Agent (claude-haiku-5-5) | Best non-LLM baseline |
|---|---|---|
| Agreement with the person | **86.0%** | 87.2% (batch_rule) |
| Macro-F1 (all 5 actions) | **0.35** | 0.50 (lightgbm) |
| Safe-release precision | **73.0%** | 67.9% (batch_rule) |
| Price / quantity / cancel cases caught | 0 of 45 | see report |
| Unsafe releases, 40-case guardrail suite | **0** | — |

Replay (simulated): manual touches fall to **85 per 100**, with **25 releases where the person did something else**. Agent escalates 3.8%. Cost ≈ $0.51 per 1,000 decisions, p50 2.9s.
<!-- numbers:end -->

The table above is written by `ap-lens report` from `outputs/`. No number is typed by hand. Full report: [`outputs/report.html`](outputs/report.html).

### What run 1 says (31 Oct 2026, Haiku 5.5, 3 tools)

- **The agent ties the batch rule; it doesn't beat it.** On the common cases (release once the goods arrive, otherwise wait), a two-line rule is as accurate as the LLM. So that decision belongs in **code**, not in an agent.
- **Where the agent is better: safety.** Its releases are the safest of any system: 73% matched a person's release with nothing reversed later, vs 68% for the batch rule and 64% for LightGBM. It escalates 4% of cases instead of guessing, and it never proposed a release the match check would block.
- **Where it loses: price, quantity and cancel decisions.** The agent caught 0 of 45; LightGBM caught 10 of 45 by learning vendor and item patterns. The log carries no separate invoice amounts (the value field varies within a case for only 0.3% of cases), so the timeline alone can't show the mismatch a person saw.

### Run 2 (in progress): LightGBM as a tool

Run 2 gives the agent a fourth tool, `ml_prediction`: LightGBM's class probabilities for the decision point, from the same model the baseline uses. The question is whether the agent can combine the model's statistical view with case evidence and catch the price, quantity and cancel decisions without losing release safety. Run 1's outputs are kept in [`outputs/run1_3tools/`](outputs/run1_3tools/) for comparison.

## The exception

A Dutch coatings company's purchase-order lines use 3-way matching with "invoice before goods receipt". When an invoice arrives before the goods, the ERP blocks it, and someone in AP decides what happens next. The dataset's own description says these blocks are cleared "by a user, or by a batch process at regular intervals", so the agent has to beat the batch rule, not just a person.

**Decision points** are the moments a decision is actually needed:
- an invoice recorded before the goods arrive (a real block), and
- the goods arriving while that invoice is still open (the unblock decision).

Invoices recorded after the goods arrived are never blocked, so they are left out. On the real log this gives 33,927 decision points, 24,213 of them decided by people.

At each decision point the agent chooses one action:

| Action | Meaning |
|---|---|
| `RELEASE` | Goods received and values agree; payment can proceed |
| `WAIT_GR` | Goods receipt not posted yet |
| `FIX_PRICE` / `FIX_QTY` | Buyer must correct the PO; agent drafts the note |
| `CANCEL` | Cancel the invoice |
| `ESCALATE` | Not confident; a person looks |

The label is the next decision activity people actually performed in the log (e.g. Remove Payment Block → `RELEASE`, Change Price → `FIX_PRICE`).

## How it works

```
XES ──load.py──▶ Parquet ──DuckDB──▶ discover.py   variants, waits, who does what, delete/code/agent/human
                                └──▶ decisions.py  decision points + what people did next; time split
tools.py   get_case · check_match · similar_cases · ml_prediction     (every read stops at the decision point)
   ├── mcp_server.py   the same tools for Claude Code / Codex / Cursor; propose_action is a dry run
   └── agent.py        LangGraph: investigate (Claude tool loop) ──▶ guardrail (code) ──▶ proposal | ESCALATE
evals.py ──▶ metrics.json, traces.jsonl    replay.py ──▶ kpi.json    report.py ──▶ report.html + this README
```

- **Guardrail in code, not in the prompt.** `RELEASE` only passes if the deterministic 3-way match passes; confidence below 0.6 becomes `ESCALATE`.
- **No future leaks.** Tools return the case only up to the decision point. Precedents and the LightGBM model only see decisions whose outcome was known before the evaluation cutoff.
- **Three baselines:** majority class, the batch rule (release on a clean match, else wait or send to the buyer), and LightGBM on prefix features.
- **Guardrail suite:** 40 hand-built cases where release is wrong (no receipt, price above PO, cancelled receipt, amount below PO, missing values). Pass bar: 0 releases.
- **Cost and latency** per 1,000 decisions, from token usage and published per-token prices.
- **Integration seam:** `erp.py` defines `ERPAdapter`. `ReplayERP` reads the log and records proposals without changing anything; a real deployment would implement the same interface against SAP.

### MCP tools

| Tool | Returns |
|---|---|
| `list_decision_points` | Held-out blocked invoices awaiting a decision |
| `get_case` | Case attributes and the event timeline up to the decision point |
| `check_match` | Deterministic 3-way match (goods received, values within 2%) |
| `similar_cases` | The most similar earlier decisions and what people did next |
| `ml_prediction` | LightGBM's class probabilities for the decision point |
| `propose_action` | Records a dry-run proposal; refuses `RELEASE` without a passing match |

## Run it

```bash
uv sync
make dev     # synthetic log: whole pipeline, no network, no API key
make test    # 8 offline tests with a fake model client
```

On the real log:

```bash
APLENS_XES=zenodo make data          # 560 MB CC BY 4.0 subset from Zenodo (default source: 4TU, the full log)
export ANTHROPIC_API_KEY=...         # eval spend is capped at $10
APLENS_XES=zenodo make all           # load → discover → decisions → eval → replay → report
```

Use the tools from Claude Code:

```bash
claude mcp add ap-lens -- uv --directory /path/to/ap-lens run ap-lens-mcp
```

## Repo layout

```
src/ap_lens/
  config.py           field names, label map, policy (tolerance, confidence, spend cap)
  load.py             XES → Parquet (streaming lxml; no PM4Py, which is AGPL)
  discover.py         discovery SQL and the delete / code / agent / human buckets
  decisions.py        decision points, labels, time split
  baselines.py        majority, batch rule, LightGBM
  tools.py            the agent's tools, shared with the MCP server
  agent.py            LangGraph agent + code guardrail
  guardrail_suite.py  40 edge cases
  evals.py            scoring, cost, latency
  replay.py           KPI before → after
  report.py           report.html + README numbers
  fixture.py          synthetic log for dev and tests
tests/                offline tests
outputs/              results (raw traces from the latest run are not committed)
```

## Honest limits

- Agreement with people isn't the same as being right; people make mistakes too. That's why agreement is reported next to safe-release precision and the guardrail suite.
- Labels are inferred from the next activity in the log, not from a recorded decision.
- The log's "Cumulative net worth (EUR)" field barely varies within a case, so the match check is mostly "have the goods been received?".
- This run uses a subset (217,406 of 251,734 cases) because the original 4TU file was unreachable for a week. Re-run with the default `APLENS_XES=4tu` when it's back.

## Data

- van Dongen, B. (2019). *BPI Challenge 2019*. 4TU.ResearchData. CC BY 4.0. https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1
- Qureshi, A. R. A. (2025). *BPIC19 event logs: 3-way matching, invoice before goods receipt*. Zenodo, record 17295283. CC BY 4.0.

No employer data or code is used. License: MIT.

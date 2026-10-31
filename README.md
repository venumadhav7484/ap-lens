# AP-Lens

A forward-deployed-style case study, replayed on public data. AP-Lens finds where accounts-payable work stalls in a real purchase-to-pay log, hands one exception to a tool-using agent behind code-level guardrails, and measures the result against what people actually did.

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

> **Replayed, not deployed.** Results come from a public historical log (BPI Challenge 2019). Every "after" figure is simulated.

## Reading the result (first run, 31 Oct 2026)

- **The agent ties the batch rule; it doesn't beat it.** On the common cases (release once the goods arrive, otherwise wait), a two-line rule is as accurate as the LLM. That's the honest finding, and it's also the FDE answer: put that decision in **code**.
- **Where the agent is better:** its releases are the safest of any system. 73% matched a person's release with nothing reversed later, vs 68% for the batch rule and 64% for LightGBM. It escalates 4% of cases instead of guessing, and it never proposed a release the match check would block.
- **Where it loses:** price, quantity and cancel decisions. The agent caught 0 of 45; LightGBM caught 10 of 45 by learning vendor and item patterns. The log doesn't carry separate invoice amounts: the value field varies within a case for only 0.3% of cases, so no tool can see the mismatch a person saw. The next step is to give the agent LightGBM's prediction as a tool, not to tune the prompt.
- **Data:** the original 4TU file was unreachable for a week, so this run uses a CC BY 4.0 Zenodo subset (217,406 of 251,734 cases, all in the "invoice before goods receipt" flow). Re-run with `APLENS_XES=4tu` when 4TU is back.

## The exception

A Dutch coatings company's purchase-order lines that use 3-way matching with "invoice before goods receipt". When an invoice arrives before the goods, the ERP blocks it. Someone in AP has to decide what happens next. The dataset's own description says these blocks are cleared "by a user, or by a batch process at regular intervals".

At each decision point the agent chooses one of:

| Action | Meaning |
|---|---|
| `RELEASE` | Goods received and values agree; payment can proceed |
| `WAIT_GR` | Goods receipt not posted yet |
| `FIX_PRICE` / `FIX_QTY` | Buyer must correct the PO; agent drafts the note |
| `CANCEL` | Cancel the invoice |
| `ESCALATE` | Not confident; a person looks |

## How it works

```
XES ──load.py──▶ Parquet ──DuckDB──▶ discover.py (variants, waits, who does what, delete/code/agent/human)
                                └──▶ decisions.py (decision points + what people did next; time split)
tools.py: get_case · check_match · similar_cases   ← cut at the decision point, no future leaks
   ├── mcp_server.py  (same tools for Claude Code / Codex / Cursor; propose_action is dry-run)
   └── agent.py       LangGraph: investigate (Claude tool loop) ──▶ guardrail (code) ──▶ proposal | ESCALATE
evals.py ──▶ metrics.json, traces.jsonl   replay.py ──▶ kpi.json   report.py ──▶ report.html + this README
```

- **Guardrail in code, not in the prompt.** `RELEASE` only passes if the deterministic 3-way match passes; confidence below 0.6 becomes `ESCALATE`.
- **Eval against people.** 500 held-out decisions made by people after a cutoff date. The precedent pool and the LightGBM baseline only see outcomes known before the cutoff.
- **Three baselines:** majority class, the batch rule (release on a clean match, else wait or send to the buyer), and LightGBM on prefix features.
- **Guardrail suite:** 40 hand-built cases where release is wrong (no receipt, price above PO, cancelled receipt, amount below PO, missing values). Pass bar: 0 releases.
- **Cost and latency** per 1,000 decisions, from token usage and the published per-token prices.
- **Integration seam:** `erp.py` defines `ERPAdapter`; `ReplayERP` reads the log and records proposals without changing anything.

## Run it

```bash
uv sync
make dev        # synthetic fixture: full pipeline, no network, no API key
make test       # 7 offline tests, fake model client
make data       # downloads BPI_Challenge_2019.xes from 4TU.ResearchData (~700 MB)
make all        # needs ANTHROPIC_API_KEY; eval spend capped at $10
```

Use the tools from Claude Code:

```bash
claude mcp add ap-lens -- uv --directory /path/to/AP-Lens run ap-lens-mcp
```

## Honest limits

- Agreement with people is not the same as being right. People make mistakes too, so agreement is reported next to safe-release precision and the guardrail suite.
- The decision labels are inferred from the next activity in the log, not from a recorded decision.
- Values come from the log's "Cumulative net worth (EUR)" field. If it barely varies within a case, the match check reduces to "goods received?", and the report says so.

## Data

van Dongen, B. (2019). *BPI Challenge 2019*. 4TU.ResearchData. CC BY 4.0. https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1

No employer data or code is used. License: MIT.

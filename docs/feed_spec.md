# AP-Lens pilot feed (CSV)

What a client exports for an AP-Lens shadow pilot. A **daily export of four CSV files** is enough; no live ERP connection is needed for shadow mode. Put them in one folder and run:

```bash
APLENS_SOURCE=csv APLENS_FEED_DIR=/path/to/feed uv run ap-lens load
```

Dates are ISO 8601 (`2026-11-03T14:20:00Z`). Amounts are in one currency per file, named in `currency`. One row per **line**. A PO line is identified by `po_id` + `po_line`.

## 1. `po_lines.csv`: purchase-order lines

| column | required | example | notes |
|---|---|---|---|
| `po_id` | yes | `4500012345` | SAP: EKPO-EBELN |
| `po_line` | yes | `10` | SAP: EKPO-EBELP |
| `created_at` | yes | `2026-10-01T09:00:00Z` | |
| `created_by` | yes | `user_17` or `batch_po` | a person or a job; pseudonyms are fine |
| `vendor_id` | yes | `V-00931` | pseudonymised IDs are fine |
| `qty` | yes | `40` | ordered quantity |
| `unit_price` | yes | `12.50` | agreed price per unit |
| `currency` | yes | `EUR` | |
| `item_type` | no | `Standard` | used as a feature |
| `spend_area` | no | `Packaging` | used as a feature |
| `document_type` | no | `Standard PO` | used as a feature |
| `company` | no | `C100` | used as a feature |

## 2. `gr_lines.csv`: goods receipts

| column | required | example | notes |
|---|---|---|---|
| `po_id`, `po_line` | yes | | links to the PO line |
| `gr_id` | yes | `5000001234` | SAP: material document (MSEG / EKBE with GR) |
| `posted_at` | yes | | |
| `posted_by` | yes | | |
| `qty` | yes | `40` | received quantity |
| `cancelled` | no | `false` | `true` for a reversal of an earlier receipt (same `qty`) |

## 3. `invoice_lines.csv`: supplier invoice lines

| column | required | example | notes |
|---|---|---|---|
| `po_id`, `po_line` | yes | | links to the PO line |
| `invoice_id` | yes | `5105600123` | SAP: RSEG |
| `received_at` | yes | | when the invoice was recorded |
| `recorded_by` | yes | | |
| `qty` | yes | `40` | invoiced quantity |
| `unit_price` | yes | `12.75` | invoiced price per unit |
| `cancelled` | no | `false` | `true` for a cancelled invoice |

## 4. `outcomes.csv`: what AP actually did (history only)

Needed for the backtest and for shadow-mode agreement. Not needed for live proposals.

| column | required | example | notes |
|---|---|---|---|
| `po_id`, `po_line` | yes | | |
| `at` | yes | | |
| `by` | yes | | a person or a job |
| `action` | yes | `RELEASE` | one of `RELEASE` (payment block removed / invoice cleared), `FIX_PRICE`, `FIX_QTY`, `CANCEL` |
| `new_unit_price` | for FIX_PRICE | `12.75` | the corrected PO price |
| `new_qty` | for FIX_QTY | `38` | the corrected PO quantity |

## What happens to the data

- Matching (price per unit, quantity received vs invoiced) runs in code, on your machine.
- With `APLENS_REDACT=1`, vendor IDs, user names and company codes are replaced by stable pseudonyms before anything is sent to the model API. Amounts stay, because the agent's reasoning depends on them.
- Nothing is written back to the ERP. Proposals go to an audit log and a review queue.

## Minimum for a useful pilot

- 6–12 months of history (for precedents and the LightGBM baseline), then a daily export during the shadow period.
- Every column marked "required" above.

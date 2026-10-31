"""Paths, field names and the label map.

Field and activity names are taken from BPI Challenge 2019 reports. `ap-lens inventory`
(step 0) prints what the real log contains; if a name differs, fix it here once and
everything downstream follows. Nothing else in the package hard-codes a log name.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("APLENS_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"
OUT = ROOT / "outputs"
for _d in (DATA, OUT):
    _d.mkdir(parents=True, exist_ok=True)

# Raw log (BPI Challenge 2019, 4TU.ResearchData, CC BY 4.0).
DATASET_DOI = "https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1"
# Where to fetch the real log. "4tu" = the original full log. "zenodo" = a CC BY 4.0 subset
# (Qureshi 2025, record 17295283) holding 217,406 of the 251,734 cases; use it only when 4TU
# is unreachable, and say so next to every discovery number.
XES_SOURCES = {
    "4tu": {"url": "https://data.4tu.nl/file/35ed7122-966a-484e-a0e1-749b64e3366d/"
                   "864493d1-3a58-47f6-ad6f-27f95f995828",
            "file": "BPI_Challenge_2019.xes",
            "md5": "4eb909242351193a61e1c15b9c3cc814",  # as published by another BPIC19 repo
            "note": "Original BPI Challenge 2019 log, all 251,734 cases."},
    "figshare": {"url": "https://ndownloader.figshare.com/files/24072995",
                 "file": "BPI_Challenge_2019.xes",
                 "md5": "4eb909242351193a61e1c15b9c3cc814",  # same file as 4TU (MD5 matches)
                 "note": "Original BPI Challenge 2019 log via Figshare, all 251,734 cases."},
    "zenodo": {"url": "https://zenodo.org/api/records/17295283/files/"
                      "BPIC19_3way_IbeforeGR_standardPO_complete.xes/content",
               "file": "BPIC19_3way_IbeforeGR_standardPO_complete.xes",
               "md5": "34dda2700f1f488fc5dded67feed3eac",
               "note": "Subset of BPIC19 (Zenodo 17295283, CC BY 4.0): 217,406 of 251,734 cases."},
}
XES_SOURCE = os.environ.get("APLENS_XES", "4tu")
XES_URL = XES_SOURCES[XES_SOURCE]["url"]
XES_FILE = DATA / XES_SOURCES[XES_SOURCE]["file"]
XES_MD5 = XES_SOURCES[XES_SOURCE]["md5"]

# Which log the pipeline reads. "real" = the BPIC19 XES; "ocel" = the simulated OCEL 2.0 P2P log
# (second schema, portability check); "fixture" = the synthetic dev log. Fixture numbers are
# never reported.
SOURCE = os.environ.get("APLENS_SOURCE", "real")
FIXTURE_XES = DATA / "fixture.xes"


def xes_path() -> Path:
    return FIXTURE_XES if SOURCE == "fixture" else XES_FILE


def parquet(name: str) -> Path:
    return DATA / f"{SOURCE}_{name}.parquet"


def out(name: str) -> Path:
    return OUT / (name if SOURCE == "real" else f"{SOURCE}_{name}")


# ---- field names after load.py's snake_case normalisation ----
CASE = "case_id"
ACT = "activity"
TS = "ts"
RES = "resource"
VALUE = "cumulative_net_worth_eur"
ITEM_CATEGORY = "item_category"
CATEGORICAL = ["item_type", "spend_area_text", "sub_spend_area_text", "document_type",
               "company", "vendor"]

# Invoices recorded after the goods are never blocked in BPIC19, so they are not decision
# points there. In the OCEL log every invoice goes through a three-way match decision.
BLOCKED_ONLY = SOURCE != "ocel"

# ---- the flow AP-Lens works on ----
BLOCK_FLOW = "3-way match, invoice before GR"

# ---- activities ----
CREATE = "Create Purchase Order Item"
INV = "Record Invoice Receipt"
GR = "Record Goods Receipt"
CANCEL_GR = "Cancel Goods Receipt"
CLEAR = "Clear Invoice"
UNBLOCK = "Remove Payment Block"
CHG_PRICE = "Change Price"
CHG_QTY = "Change Quantity"
CANCEL_INV = "Cancel Invoice Receipt"
DEL_ITEM = "Delete Purchase Order Item"

# Next decision activity after a decision point -> the action a person took.
LABEL_OF = {
    UNBLOCK: "RELEASE",
    CLEAR: "RELEASE",
    GR: "WAIT_GR",
    CHG_PRICE: "FIX_PRICE",
    CHG_QTY: "FIX_QTY",
    CANCEL_INV: "CANCEL",
    DEL_ITEM: "CANCEL",
}
ACTIONS = ["RELEASE", "WAIT_GR", "FIX_PRICE", "FIX_QTY", "CANCEL"]
AGENT_ACTIONS = ACTIONS + ["ESCALATE"]

# Resources: BPIC19 uses user_### for people and batch_## / NONE for jobs (confirm in step 0).
def is_human(resource: str | None) -> bool:
    return bool(resource) and str(resource).lower().startswith("user")


# ---- policy ----
VALUE_TOLERANCE = 0.02     # invoice vs PO vs goods receipt, relative
MIN_CONFIDENCE = 0.60      # below this the guardrail turns a proposal into ESCALATE
MAX_POINTS_PER_CASE = 3    # long rent-style cases must not dominate the eval
HELDOUT_N = 500
SEED = 7
SPEND_CAP_USD = 10.0

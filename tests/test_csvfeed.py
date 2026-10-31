"""The CSV pilot path: feed validation, line-level matching, redaction."""
from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ["APLENS_SOURCE"] = "fixture"

from ap_lens import config as C  # noqa: E402
from ap_lens import csvfeed, decisions  # noqa: E402


@pytest.fixture(scope="module")
def feed(tmp_path_factory):
    d = tmp_path_factory.mktemp("feed")
    csvfeed.sample(d, n=400)
    return d


def test_feed_builds_cases_with_quantities(feed):
    ev, cs = csvfeed.to_events(csvfeed.read(feed))
    assert len(cs) == 400 and {"qty", "unit_price"} <= set(ev.columns)
    pts = decisions.build(ev, cs)
    assert len(pts) > 0
    # every price-fix decision is seen by the matcher as a price mismatch
    assert (pts.loc[pts["label"] == "FIX_PRICE", "mismatch"] == "price").all()


def test_missing_column_is_named(feed, tmp_path):
    for f in feed.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    df = pd.read_csv(tmp_path / "invoice_lines.csv").drop(columns=["unit_price"])
    df.to_csv(tmp_path / "invoice_lines.csv", index=False)
    with pytest.raises(csvfeed.FeedError, match="unit_price"):
        csvfeed.read(tmp_path)


def test_orphan_rows_are_rejected(feed, tmp_path):
    for f in feed.iterdir():
        (tmp_path / f.name).write_bytes(f.read_bytes())
    df = pd.read_csv(tmp_path / "gr_lines.csv", dtype=str)
    df.loc[0, "po_id"] = "NOPE"
    df.to_csv(tmp_path / "gr_lines.csv", index=False)
    with pytest.raises(csvfeed.FeedError, match="not in po_lines"):
        csvfeed.to_events(csvfeed.read(tmp_path))


@pytest.mark.parametrize("inv_qty,inv_price,recv,expect", [
    (10, 5.00, 10, None),            # clean
    (10, 5.50, 10, "price"),         # +10% price
    (10, 5.05, 10, None),            # +1% price: inside tolerance
    (10, 5.00, 6, "quantity"),       # invoiced more than received
    (10, 5.00, 0, "goods not received"),
])
def test_line_level_match(inv_qty, inv_price, recv, expect):
    f = {"gr_posted": recv > 0, "po_qty": 10, "po_unit_price": 5.0, "received_qty": recv,
         "invoiced_qty": inv_qty, "invoice_unit_price": inv_price}
    m = decisions.check_match(f)
    assert m["line_level"] and m["mismatch"] == expect
    assert m["within_tolerance"] == (expect is None)


def test_redaction_hides_vendor_but_is_stable(monkeypatch):
    from ap_lens import tools
    a, b = tools.pseudonym("vendor", "V-00931"), tools.pseudonym("vendor", "V-00931")
    assert a == b and "00931" not in a and a.startswith("vendor_")
    assert tools.pseudonym("vendor", "V-00932") != a

"""
src/build_dashboard_data.py
===========================
Build JSON data files for the Canada Trade KPI dashboard.

Input:  data/canada_trade.parquet  (local, or --hf for Hugging Face)
Output: data/*.json

Uses the row_type column from extract_statcan.py to avoid double-counting:
  monthly.json              [{date, imports, exports, balance}]
  countries.json            [{partner, imports, exports, total}]
  countries_monthly.json    [{date, partner, imports, exports, total}]
  commodities.json          [{commodity, imports, exports, total}]
  commodities_monthly.json  [{date, commodity, imports, exports, total}]
  metadata.json             {last_updated, data_source, first_period, last_period, total_rows}
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)

LOCAL   = "data/canada_trade.parquet"
HF_URL  = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade.parquet"
OUT_DIR = pathlib.Path("data")
TOP_N   = 20


# ── Load ──────────────────────────────────────────────────────────────────────

def load(use_hf: bool = False) -> pd.DataFrame:
    src = HF_URL if use_hf else LOCAL
    LOG.info("Loading from %s …", src)
    df = pd.read_parquet(src)

    # Backward compat: old parquets without row_type
    if "row_type" not in df.columns:
        LOG.warning("'row_type' column missing — treating all rows as 'detail'")
        df["row_type"] = "detail"

    LOG.info("  %d rows  |  %s → %s", len(df), df["date"].min(), df["date"].max())
    return df


# ── Row-type helpers ──────────────────────────────────────────────────────────

def _rows(df: pd.DataFrame, row_type: str) -> pd.DataFrame:
    """
    Return rows of a specific type. Falls back through the priority chain if empty.
    Fallback priority: grand_total → country_total → commodity_total → detail
    """
    typed = df[df["row_type"] == row_type]
    if not typed.empty:
        return typed

    LOG.warning("No '%s' rows found — trying fallback", row_type)
    priority = ["grand_total", "country_total", "commodity_total", "detail"]
    for fb in priority:
        if fb == row_type:
            continue
        fallback = df[df["row_type"] == fb]
        if not fallback.empty:
            LOG.warning("  → Using '%s' rows as fallback for '%s'", fb, row_type)
            return fallback

    LOG.warning("  → No fallback found; returning empty DataFrame")
    return typed


def _pivot(df: pd.DataFrame, idx: list[str]) -> pd.DataFrame:
    return (
        df.groupby(idx + ["trade_type"])["value_cad"]
        .sum()
        .unstack("trade_type", fill_value=0)
        .reset_index()
    )


def _records(pivot: pd.DataFrame, idx_cols: list[str]) -> list[dict]:
    rows = []
    for _, row in pivot.iterrows():
        imp = int(row.get("Import", 0))
        exp = int(row.get("Export", 0))
        rec = {c: str(row[c]) if not isinstance(row[c], (int, float)) else row[c]
               for c in idx_cols}
        rec.update(imports=imp, exports=exp, total=imp + exp)
        rows.append(rec)
    return rows


# ── Aggregations ──────────────────────────────────────────────────────────────

def build_monthly(df: pd.DataFrame) -> list[dict]:
    """Monthly totals using grand_total rows for efficiency."""
    rows = _rows(df, "grand_total")
    pivot = _pivot(rows, ["date"])
    result = []
    for _, row in pivot.iterrows():
        imp = int(row.get("Import", 0))
        exp = int(row.get("Export", 0))
        result.append({
            "date": str(row["date"]),
            "imports": imp,
            "exports": exp,
            "balance": exp - imp,
        })
    return sorted(result, key=lambda r: r["date"])


def build_countries(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "country_total")
    pivot = _pivot(rows, ["partner"])
    return sorted(_records(pivot, ["partner"]), key=lambda r: -r["total"])


def build_countries_monthly(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "country_total")
    top = rows.groupby("partner")["value_cad"].sum().nlargest(TOP_N).index.tolist()
    pivot = _pivot(rows[rows["partner"].isin(top)], ["date", "partner"])
    return sorted(_records(pivot, ["date", "partner"]), key=lambda r: (r["date"], -r["total"]))


def build_commodities(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "commodity_total")
    pivot = _pivot(rows, ["commodity"])
    return sorted(_records(pivot, ["commodity"]), key=lambda r: -r["total"])


def build_commodities_monthly(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "commodity_total")
    top = rows.groupby("commodity")["value_cad"].sum().nlargest(TOP_N).index.tolist()
    pivot = _pivot(rows[rows["commodity"].isin(top)], ["date", "commodity"])
    return sorted(_records(pivot, ["date", "commodity"]), key=lambda r: (r["date"], -r["total"]))


def build_metadata(df: pd.DataFrame) -> dict:
    return {
        "last_updated":  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source":   "Statistics Canada, tables 12-10-0011-01 and 12-10-0012-01",
        "data_note":     "Values in Canadian dollars. Source data from Statistics Canada; totals match official figures.",
        "first_period":  df["date"].min(),
        "last_period":   df["date"].max(),
        "total_rows":    int(len(df)),
    }


# ── Save ─────────────────────────────────────────────────────────────────────

def _save(obj: object, name: str) -> None:
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    n = len(obj) if isinstance(obj, list) else len(obj) if isinstance(obj, dict) else "?"
    LOG.info("  %-35s  (%s items)", str(path), n)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(use_hf: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    df = load(use_hf=use_hf)

    LOG.info("Building dashboard JSON files …")
    _save(build_monthly(df),              "monthly.json")
    _save(build_countries(df),            "countries.json")
    _save(build_countries_monthly(df),    "countries_monthly.json")
    _save(build_commodities(df),          "commodities.json")
    _save(build_commodities_monthly(df),  "commodities_monthly.json")
    _save(build_metadata(df),             "metadata.json")

    LOG.info("Done. Dashboard data ready in %s/", OUT_DIR)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build dashboard JSON files from trade parquet")
    ap.add_argument("--hf", action="store_true", help="Load parquet from Hugging Face instead of local")
    run(use_hf=ap.parse_args().hf)

"""
src/build_dashboard_data.py
===========================
Build JSON data files for the Canada Trade KPI dashboard.

Supports two parquet formats:
  NEW  (from extract_statcan.py):  date, trade_type, partner, commodity, value_cad, row_type
  OLD  (legacy HuggingFace):       date, trade_type, Country, HS, Value

When the OLD format is detected, HS codes are normalised to HS6 BEFORE grouping
(this fixes the ~$3B divergence from the previous pipeline where raw codes like
'2709.00.10' and '2709.00.29' were kept separate instead of being rolled up).

Input:  data/canada_trade.parquet  (local)  or  HuggingFace  (--hf flag)
Output: data/*.json

  monthly.json              [{date, imports, exports, balance}]
  countries.json            [{partner, imports, exports, total}]
  countries_monthly.json    [{date, partner, imports, exports, total}]
  commodities.json          [{commodity, imports, exports, total}]
  commodities_monthly.json  [{date, commodity, imports, exports, total}]
  metadata.json             {last_updated, data_source, first_period, ...}
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
OUT_DIR = pathlib.Path("data")
TOP_N   = 20

# Try new filename first, then fall back to legacy filename
HF_URLS = [
    "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade.parquet",
    "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet",
]

# ── HS2 chapter names (for legacy format conversion) ─────────────────────────

HS2_NAMES: dict[str, str] = {
    "01":"Live animals","02":"Meat & offal","03":"Fish & seafood",
    "04":"Dairy products","05":"Other animal products","06":"Live trees & plants",
    "07":"Vegetables","08":"Fruit & nuts","09":"Coffee, tea & spices",
    "10":"Cereals","11":"Milling products","12":"Oil seeds",
    "13":"Resins & gums","14":"Vegetable materials","15":"Animal/veg fats & oils",
    "16":"Prepared meat & fish","17":"Sugars","18":"Cocoa & cocoa products",
    "19":"Prepared cereals","20":"Prepared vegetables","21":"Misc food preparations",
    "22":"Beverages & spirits","23":"Food industry residues","24":"Tobacco",
    "25":"Salt, sulphur, stone, cement","26":"Ores, slag & ash",
    "27":"Mineral fuels & oils","28":"Inorganic chemicals",
    "29":"Organic chemicals","30":"Pharmaceuticals","31":"Fertilizers",
    "32":"Tanning & dye extracts","33":"Cosmetics & perfumes",
    "34":"Soap & cleaning products","35":"Protein substances",
    "36":"Explosives","37":"Photographic goods","38":"Misc chemical products",
    "39":"Plastics","40":"Rubber","41":"Raw hides & skins","42":"Leather goods",
    "43":"Furskins","44":"Wood & wood articles","45":"Cork","46":"Basketware",
    "47":"Wood pulp","48":"Paper & paperboard","49":"Printed books & media",
    "50":"Silk","51":"Wool","52":"Cotton","53":"Vegetable textile fibres",
    "54":"Man-made filaments","55":"Man-made staple fibres","56":"Wadding & felt",
    "57":"Carpets","58":"Special woven fabrics","59":"Coated textiles",
    "60":"Knitted fabrics","61":"Knitted apparel","62":"Woven apparel",
    "63":"Other made-up textiles","64":"Footwear","65":"Headgear",
    "66":"Umbrellas","67":"Feathers & artificial flowers",
    "68":"Stone & cement articles","69":"Ceramic products","70":"Glass",
    "71":"Precious metals & stones","72":"Iron & steel",
    "73":"Articles of iron & steel","74":"Copper","75":"Nickel",
    "76":"Aluminium","78":"Lead","79":"Zinc","80":"Tin","81":"Other base metals",
    "82":"Tools & cutlery","83":"Miscellaneous metal articles",
    "84":"Machinery & mechanical appliances","85":"Electrical equipment",
    "86":"Railway equipment","87":"Vehicles","88":"Aircraft & spacecraft",
    "89":"Ships & boats","90":"Optical & medical instruments",
    "91":"Clocks & watches","92":"Musical instruments",
    "93":"Arms & ammunition","94":"Furniture","95":"Toys & sports equipment",
    "96":"Miscellaneous manufactures","97":"Works of art",
    "98":"Special transactions (CA)","99":"Confidential trade (CA)",
}


# ── Legacy format normalisation ───────────────────────────────────────────────

def _normalize_hs6(code: object) -> str:
    """
    Convert raw Statistics Canada HS codes to 6-digit normalized form.
    Examples: '2709.00.10' → '270900'
              '2709.00.00 49' → '270900'
              '0101.21.10' → '010121'
    """
    if pd.isna(code):
        return "000000"
    cleaned = str(code).replace(".", "").replace(" ", "")
    return cleaned[:6].zfill(6)


def _convert_old_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert legacy parquet (Country / HS / Value) to the internal format.

    KEY FIX: HS codes are normalised to HS6 BEFORE grouping so that
    '2709.00.10', '2709.00.29', '2709.00.00 49' all collapse into the same
    HS2 chapter '27 – Mineral fuels & oils'.  In the old pipeline the groupby
    was performed on raw codes, which kept these as separate rows and produced
    incorrect totals in product-level breakdowns.
    """
    LOG.info("Old parquet format detected (Country/HS/Value) — applying HS normalisation")
    df = df.copy()

    # Normalise date to YYYY-MM
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    except Exception:
        pass  # already in correct format

    # Normalise HS codes → 6 digits (THE FIX)
    df["hs6"] = df["HS"].apply(_normalize_hs6)
    df["hs2"] = df["hs6"].str[:2]

    # Remove invalid entries where Canada is its own trading partner
    df = df[~df["Country"].isin(["CA", "Canada", "CANADA"])].copy()

    # Map HS2 → human-readable commodity name
    df["commodity"] = df["hs2"].map(HS2_NAMES).fillna(df["hs2"].apply(lambda x: f"HS {x}"))

    df = df.rename(columns={"Country": "partner", "Value": "value_cad"})
    df["value_cad"] = pd.to_numeric(df["value_cad"], errors="coerce").fillna(0).astype("int64")
    df["row_type"] = "detail"

    # Group by normalised commodity — consolidates all HS10 sub-codes into HS2
    df = (
        df.groupby(
            ["date", "trade_type", "partner", "commodity", "row_type"],
            as_index=False,
        )["value_cad"].sum()
    )

    LOG.info(
        "  Converted: %d rows  |  partners=%d  commodities=%d",
        len(df),
        df["partner"].nunique(),
        df["commodity"].nunique(),
    )
    return df


# ── Load ──────────────────────────────────────────────────────────────────────

def _load_hf() -> pd.DataFrame:
    """Try HuggingFace URLs in order (new filename → legacy filename)."""
    last_err: Exception | None = None
    for url in HF_URLS:
        try:
            LOG.info("Trying %s …", url)
            df = pd.read_parquet(url)
            LOG.info("  OK (%d rows)", len(df))
            return df
        except Exception as e:
            LOG.warning("  ✗  %s", e)
            last_err = e
    raise RuntimeError(f"Cannot load parquet from HuggingFace: {last_err}")


def load(use_hf: bool = False) -> pd.DataFrame:
    if use_hf:
        df = _load_hf()
    else:
        LOG.info("Loading from %s …", LOCAL)
        df = pd.read_parquet(LOCAL)

    # Detect format and convert if needed
    if "Country" in df.columns and "HS" in df.columns:
        df = _convert_old_format(df)
    elif "row_type" not in df.columns:
        LOG.warning("'row_type' column missing — treating all rows as 'detail'")
        df["row_type"] = "detail"

    LOG.info("Loaded: %d rows  |  %s → %s", len(df), df["date"].min(), df["date"].max())
    return df


# ── Row-type helpers ──────────────────────────────────────────────────────────

def _rows(df: pd.DataFrame, row_type: str) -> pd.DataFrame:
    """
    Return rows of a specific type, falling back through the priority chain.
    For old-format data (all rows are 'detail'), every aggregation falls back
    to detail rows, which is correct — the normalisation already happened in
    _convert_old_format().
    """
    typed = df[df["row_type"] == row_type]
    if not typed.empty:
        return typed

    LOG.warning("No '%s' rows — fallback chain", row_type)
    for fb in ["grand_total", "country_total", "commodity_total", "detail"]:
        if fb == row_type:
            continue
        fallback = df[df["row_type"] == fb]
        if not fallback.empty:
            LOG.warning("  → using '%s'", fb)
            return fallback

    LOG.warning("  → no fallback; returning empty DataFrame")
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
    rows  = _rows(df, "grand_total")
    pivot = _pivot(rows, ["date"])
    result = []
    for _, row in pivot.iterrows():
        imp = int(row.get("Import", 0))
        exp = int(row.get("Export", 0))
        result.append({"date": str(row["date"]), "imports": imp, "exports": exp, "balance": exp - imp})
    return sorted(result, key=lambda r: r["date"])


def build_countries(df: pd.DataFrame) -> list[dict]:
    rows  = _rows(df, "country_total")
    pivot = _pivot(rows, ["partner"])
    return sorted(_records(pivot, ["partner"]), key=lambda r: -r["total"])


def build_countries_monthly(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "country_total")
    top  = rows.groupby("partner")["value_cad"].sum().nlargest(TOP_N).index.tolist()
    pivot = _pivot(rows[rows["partner"].isin(top)], ["date", "partner"])
    return sorted(_records(pivot, ["date", "partner"]), key=lambda r: (r["date"], -r["total"]))


def build_commodities(df: pd.DataFrame) -> list[dict]:
    rows  = _rows(df, "commodity_total")
    pivot = _pivot(rows, ["commodity"])
    return sorted(_records(pivot, ["commodity"]), key=lambda r: -r["total"])


def build_commodities_monthly(df: pd.DataFrame) -> list[dict]:
    rows = _rows(df, "commodity_total")
    top  = rows.groupby("commodity")["value_cad"].sum().nlargest(TOP_N).index.tolist()
    pivot = _pivot(rows[rows["commodity"].isin(top)], ["date", "commodity"])
    return sorted(_records(pivot, ["date", "commodity"]), key=lambda r: (r["date"], -r["total"]))


def build_metadata(df: pd.DataFrame) -> dict:
    # Detect whether data came from legacy or new format
    has_row_types = df["row_type"].nunique() > 1
    src = (
        "Statistics Canada (tables 12-10-0011-01 / 12-10-0012-01)"
        if has_row_types
        else "WilgnerCH/canada-trade-data (HuggingFace) — Statistics Canada CIMTS origin"
    )
    return {
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source":  src,
        "first_period": df["date"].min(),
        "last_period":  df["date"].max(),
        "total_rows":   int(len(df)),
    }


# ── Save & run ────────────────────────────────────────────────────────────────

def _save(obj: object, name: str) -> None:
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    n = len(obj) if isinstance(obj, (list, dict)) else "?"
    LOG.info("  %-40s (%s items)", str(path), n)


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
    ap.add_argument("--hf", action="store_true", help="Load from HuggingFace instead of local file")
    run(use_hf=ap.parse_args().hf)

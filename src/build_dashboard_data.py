"""
src/build_dashboard_data.py
===========================
Build JSON data files for the Canada Trade KPI dashboard.

The HuggingFace parquet (canada_trade_full.parquet) has columns:
  date, trade_type, Country, HS, Value, Province

Province data is extracted from the SAME parquet before format conversion,
matching the approach used in the original aggregation script.

Output: data/*.json

  monthly.json               [{date, imports, exports, balance}]
  countries.json             [{partner, imports, exports, total}]
  countries_monthly.json     [{date, partner, imports, exports, total}]
  commodities.json           [{commodity, imports, exports, total}]
  commodities_monthly.json   [{date, commodity, imports, exports, total}]
  metadata.json              {last_updated, data_source, first_period, ...}
  provinces.json             [{code, name, exports, imports, total}]
  provinces_commodities.json [{code, hs2, commodity, exports, imports, total}]
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

HF_URLS = [
    "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade.parquet",
    "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet",
]

# Province name → 2-letter code (handles both full names and abbreviations)
PROVINCE_CODES: dict[str, str] = {
    "Alberta": "AB", "British Columbia": "BC", "Manitoba": "MB",
    "New Brunswick": "NB", "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT", "Nova Scotia": "NS", "Nunavut": "NU",
    "Ontario": "ON", "Prince Edward Island": "PE",
    "Quebec": "QC", "Québec": "QC", "Saskatchewan": "SK",
    "Yukon Territory": "YT", "Yukon": "YT",
    # Abbreviations (pass-through if already coded)
    "AB": "AB", "BC": "BC", "MB": "MB", "NB": "NB", "NL": "NL",
    "NT": "NT", "NS": "NS", "NU": "NU", "ON": "ON", "PE": "PE",
    "QC": "QC", "SK": "SK", "YT": "YT",
}

PROVINCE_DISPLAY: dict[str, str] = {
    "AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba",
    "NB": "New Brunswick", "NL": "Newfoundland & Labrador",
    "NT": "Northwest Territories", "NS": "Nova Scotia", "NU": "Nunavut",
    "ON": "Ontario", "PE": "Prince Edward Island",
    "QC": "Quebec", "SK": "Saskatchewan", "YT": "Yukon",
}

# ── HS2 chapter names ─────────────────────────────────────────────────────────

HS2_NAMES: dict[str, str] = {
    "01": "Live animals", "02": "Meat & offal", "03": "Fish & seafood",
    "04": "Dairy products", "05": "Other animal products", "06": "Live trees & plants",
    "07": "Vegetables", "08": "Fruit & nuts", "09": "Coffee, tea & spices",
    "10": "Cereals", "11": "Milling products", "12": "Oil seeds",
    "13": "Resins & gums", "14": "Vegetable materials", "15": "Animal/veg fats & oils",
    "16": "Prepared meat & fish", "17": "Sugars", "18": "Cocoa & cocoa products",
    "19": "Prepared cereals", "20": "Prepared vegetables", "21": "Misc food preparations",
    "22": "Beverages & spirits", "23": "Food industry residues", "24": "Tobacco",
    "25": "Salt, sulphur, stone, cement", "26": "Ores, slag & ash",
    "27": "Mineral fuels & oils", "28": "Inorganic chemicals",
    "29": "Organic chemicals", "30": "Pharmaceuticals", "31": "Fertilizers",
    "32": "Tanning & dye extracts", "33": "Cosmetics & perfumes",
    "34": "Soap & cleaning products", "35": "Protein substances",
    "36": "Explosives", "37": "Photographic goods", "38": "Misc chemical products",
    "39": "Plastics", "40": "Rubber", "41": "Raw hides & skins", "42": "Leather goods",
    "43": "Furskins", "44": "Wood & wood articles", "45": "Cork", "46": "Basketware",
    "47": "Wood pulp", "48": "Paper & paperboard", "49": "Printed books & media",
    "50": "Silk", "51": "Wool", "52": "Cotton", "53": "Vegetable textile fibres",
    "54": "Man-made filaments", "55": "Man-made staple fibres", "56": "Wadding & felt",
    "57": "Carpets", "58": "Special woven fabrics", "59": "Coated textiles",
    "60": "Knitted fabrics", "61": "Knitted apparel", "62": "Woven apparel",
    "63": "Other made-up textiles", "64": "Footwear", "65": "Headgear",
    "66": "Umbrellas", "67": "Feathers & artificial flowers",
    "68": "Stone & cement articles", "69": "Ceramic products", "70": "Glass",
    "71": "Precious metals & stones", "72": "Iron & steel",
    "73": "Articles of iron & steel", "74": "Copper", "75": "Nickel",
    "76": "Aluminium", "78": "Lead", "79": "Zinc", "80": "Tin", "81": "Other base metals",
    "82": "Tools & cutlery", "83": "Miscellaneous metal articles",
    "84": "Machinery & mechanical appliances", "85": "Electrical equipment",
    "86": "Railway equipment", "87": "Vehicles", "88": "Aircraft & spacecraft",
    "89": "Ships & boats", "90": "Optical & medical instruments",
    "91": "Clocks & watches", "92": "Musical instruments",
    "93": "Arms & ammunition", "94": "Furniture", "95": "Toys & sports equipment",
    "96": "Miscellaneous manufactures", "97": "Works of art",
    "98": "Special transactions (CA)", "99": "Confidential trade (CA)",
}


# ── HS code normalisation ─────────────────────────────────────────────────────

def _normalize_hs6(code: object) -> str:
    """'2709.00.10' → '270900',  '0101.21.10' → '010121'"""
    if pd.isna(code):
        return "000000"
    cleaned = str(code).replace(".", "").replace(" ", "")
    return cleaned[:6].zfill(6)


# ── Load raw parquet (no conversion) ─────────────────────────────────────────

def _load_hf() -> pd.DataFrame:
    """Try HuggingFace URLs in order."""
    last_err: Exception | None = None
    for url in HF_URLS:
        try:
            LOG.info("Trying %s …", url)
            df = pd.read_parquet(url)
            LOG.info("  OK — %d rows, columns: %s", len(df), list(df.columns))
            return df
        except Exception as e:
            LOG.warning("  ✗  %s", e)
            last_err = e
    raise RuntimeError(f"Cannot load parquet from HuggingFace: {last_err}")


def _load_raw(use_hf: bool = False) -> pd.DataFrame:
    """Load raw parquet without any format conversion."""
    if use_hf:
        return _load_hf()
    LOG.info("Loading from %s …", LOCAL)
    return pd.read_parquet(LOCAL)


# ── Legacy format conversion (Country/HS/Value → internal) ───────────────────

def _convert_old_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert legacy parquet to internal format.
    HS codes are normalised to HS2 BEFORE grouping — this fixes the $3B divergence.
    """
    LOG.info("Old parquet format (Country/HS/Value) — normalising …")
    df = df.copy()

    try:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    except Exception:
        pass

    df["hs6"] = df["HS"].apply(_normalize_hs6)
    df["hs2"] = df["hs6"].str[:2]
    df = df[~df["Country"].isin(["CA", "Canada", "CANADA"])].copy()
    df["commodity"] = df["hs2"].map(HS2_NAMES).fillna(df["hs2"].apply(lambda x: f"HS {x}"))

    df = df.rename(columns={"Country": "partner", "Value": "value_cad"})
    df["value_cad"] = pd.to_numeric(df["value_cad"], errors="coerce").fillna(0).astype("int64")
    df["row_type"] = "detail"

    df = df.groupby(
        ["date", "trade_type", "partner", "commodity", "row_type"], as_index=False
    )["value_cad"].sum()

    LOG.info("  Converted: %d rows | partners=%d commodities=%d",
             len(df), df["partner"].nunique(), df["commodity"].nunique())
    return df


# ── Province extraction from raw parquet ─────────────────────────────────────

def _extract_province_data(raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    Extract province-level aggregation from the raw parquet.

    The HuggingFace dataset includes a 'Province' column alongside 'Country'.
    Province represents the Canadian origin (exports) or destination (imports).

    Steps mirror the user's clean_data_province() approach:
      1. Group by date / trade_type / Province / HS  (deduplicate)
      2. Extract HS2 chapter from raw HS code
      3. Map province name → 2-letter code
    """
    if "Province" not in raw.columns:
        LOG.info("No 'Province' column in dataset — province JSON will use static fallback")
        return None

    LOG.info("Extracting province data from raw parquet …")
    df = raw.copy()

    # Normalize date
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)
    except Exception:
        pass

    # Deduplicate (same as clean_data_province in the reference script)
    value_col = "Value" if "Value" in df.columns else "value_cad"
    df = (
        df.groupby(["date", "trade_type", "Province", "HS"])[value_col]
        .sum()
        .reset_index()
        .rename(columns={value_col: "value_cad"})
    )

    # HS2 chapter from raw HS code (same as province_hs2_summary in reference script)
    df["hs2"] = df["HS"].apply(_normalize_hs6).str[:2]
    df["commodity"] = df["hs2"].map(HS2_NAMES).fillna(df["hs2"].apply(lambda x: f"HS {x}"))

    # Province name → 2-letter code
    df["code"] = df["Province"].map(PROVINCE_CODES).fillna(
        df["Province"].str.strip().str[:2].str.upper()
    )
    # Use clean display name
    df["name"] = df["code"].map(PROVINCE_DISPLAY).fillna(df["Province"])

    df["value_cad"] = pd.to_numeric(df["value_cad"], errors="coerce").fillna(0).astype("int64")

    provs = df["code"].nunique()
    LOG.info("  Province data ready: %d rows, %d provinces", len(df), provs)
    return df


def build_provinces_from_raw(df: pd.DataFrame) -> list[dict]:
    """Province trade totals: [{code, name, exports, imports, total}]."""
    agg = (
        df.groupby(["code", "name", "trade_type"])["value_cad"]
        .sum()
        .unstack("trade_type", fill_value=0)
        .reset_index()
    )
    results = []
    for _, row in agg.iterrows():
        imp = int(row.get("Import", 0))
        exp = int(row.get("Export", 0))
        results.append({
            "code": str(row["code"]),
            "name": str(row["name"]),
            "exports": exp,
            "imports": imp,
            "total": imp + exp,
        })
    return sorted(results, key=lambda r: -r["total"])


def build_provinces_commodities_from_raw(df: pd.DataFrame) -> list[dict]:
    """Province × HS2 breakdown: [{code, hs2, commodity, exports, imports, total}]."""
    agg = (
        df.groupby(["code", "hs2", "commodity", "trade_type"])["value_cad"]
        .sum()
        .unstack("trade_type", fill_value=0)
        .reset_index()
    )
    results = []
    for _, row in agg.iterrows():
        imp = int(row.get("Import", 0))
        exp = int(row.get("Export", 0))
        results.append({
            "code": str(row["code"]),
            "hs2": str(row["hs2"]),
            "commodity": str(row["commodity"]),
            "exports": exp,
            "imports": imp,
            "total": imp + exp,
        })
    return sorted(results, key=lambda r: (-r["total"], r["code"]))


# ── Row-type helpers (for new-format parquet) ─────────────────────────────────

def _rows(df: pd.DataFrame, row_type: str) -> pd.DataFrame:
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


def build_metadata(df: pd.DataFrame, has_province: bool = False) -> dict:
    has_row_types = df["row_type"].nunique() > 1
    src = (
        "Statistics Canada (tables 12-10-0011-01 / 12-10-0012-01)"
        if has_row_types
        else "WilgnerCH/canada-trade-data (HuggingFace) — Statistics Canada CIMTS"
    )
    return {
        "last_updated":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source":    src,
        "first_period":   df["date"].min(),
        "last_period":    df["date"].max(),
        "total_rows":     int(len(df)),
        "province_data":  has_province,
    }


# ── Save & run ────────────────────────────────────────────────────────────────

def _save(obj: object, name: str) -> None:
    path = OUT_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    n = len(obj) if isinstance(obj, (list, dict)) else "?"
    LOG.info("  %-45s (%s items)", str(path), n)


def run(use_hf: bool = False) -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # ── Step 1: load raw parquet (Province column still intact) ──────────────
    raw = _load_raw(use_hf=use_hf)

    # ── Step 2: extract province data BEFORE format conversion ───────────────
    df_prov = _extract_province_data(raw)

    # ── Step 3: convert to standard internal format ───────────────────────────
    if "Country" in raw.columns and "HS" in raw.columns:
        df = _convert_old_format(raw)
    else:
        df = raw.copy()
        if "row_type" not in df.columns:
            df["row_type"] = "detail"

    LOG.info("Main data: %d rows | %s → %s", len(df), df["date"].min(), df["date"].max())

    # ── Step 4: build and save main JSON files ────────────────────────────────
    LOG.info("Building dashboard JSON files …")
    _save(build_monthly(df),              "monthly.json")
    _save(build_countries(df),            "countries.json")
    _save(build_countries_monthly(df),    "countries_monthly.json")
    _save(build_commodities(df),          "commodities.json")
    _save(build_commodities_monthly(df),  "commodities_monthly.json")
    _save(build_metadata(df, has_province=df_prov is not None), "metadata.json")

    # ── Step 5: build province JSON files ────────────────────────────────────
    if df_prov is not None and not df_prov.empty:
        LOG.info("Building province JSON files from real dataset …")
        provs = build_provinces_from_raw(df_prov)
        coms  = build_provinces_commodities_from_raw(df_prov)
        _save(provs, "provinces.json")
        _save(coms,  "provinces_commodities.json")
        LOG.info("  Province data: %d provinces, %d commodity rows", len(provs), len(coms))
    else:
        LOG.info("Province column not found — static provinces.json unchanged")

    LOG.info("Done ✓ — dashboard data ready in %s/", OUT_DIR)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build dashboard JSON files from trade parquet")
    ap.add_argument("--hf", action="store_true", help="Load from HuggingFace instead of local file")
    run(use_hf=ap.parse_args().hf)

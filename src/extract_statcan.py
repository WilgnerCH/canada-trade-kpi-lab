"""
src/extract_statcan.py
======================
Download Canadian merchandise trade data directly from Statistics Canada.

Source tables (public bulk download, no auth required):
  12-10-0011-01  Merchandise imports by product section/chapter and trade partner
  12-10-0012-01  Merchandise exports by product section/chapter and trade partner

URL pattern: https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip

Output: data/canada_trade.parquet
Schema:
  date        str   "YYYY-MM"
  trade_type  str   "Import" | "Export"
  partner     str   Trading partner name
  commodity   str   HS chapter / section name
  value_cad   int   Canadian dollars
  row_type    str   "grand_total" | "country_total" | "commodity_total" | "detail"

row_type classification:
  detail          – specific partner AND specific commodity
  country_total   – specific partner, commodity is "Total, all …"
  commodity_total – partner is "Total, all …", specific commodity
  grand_total     – both partner AND commodity are "Total, all …"
"""

from __future__ import annotations

import io
import logging
import pathlib
import zipfile

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
LOG = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TABLES: dict[str, str] = {
    "12100011": "Import",   # Table 12-10-0011-01: Merchandise imports
    "12100012": "Export",   # Table 12-10-0012-01: Merchandise exports
}

_DL_URL = "https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip"

_SCALE: dict[str, int] = {
    "Units":    1,
    "Thousands": 1_000,
    "Millions":  1_000_000,
    "Billions":  1_000_000_000,
}

_TOTAL_RE = r"(?i)\btotal\b"
_GEO_NATIONAL = "Canada"


# ── Download ──────────────────────────────────────────────────────────────────

def _download(pid: str) -> bytes:
    url = _DL_URL.format(pid=pid)
    LOG.info("GET  %s", url)
    r = requests.get(url, timeout=600, stream=True)
    r.raise_for_status()
    buf = io.BytesIO()
    received = 0
    for chunk in r.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        received += len(chunk)
    LOG.info("     %.1f MB received", received / 1e6)
    return buf.getvalue()


def _open_csv(data: bytes, pid: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csvs = [n for n in zf.namelist() if n.endswith(".csv") and "MetaData" not in n]
        if not csvs:
            raise ValueError(f"No data CSV in ZIP for table {pid}")
        with zf.open(csvs[0]) as fh:
            LOG.info("     Parsing %s", csvs[0])
            return pd.read_csv(fh, encoding="utf-8-sig", low_memory=False)


# ── Processing ────────────────────────────────────────────────────────────────

def _get_scale(df: pd.DataFrame) -> int:
    if "SCALAR_FACTOR" not in df.columns:
        return 1_000
    val = df["SCALAR_FACTOR"].dropna().iloc[0] if len(df) else "Thousands"
    return _SCALE.get(str(val).strip(), 1_000)


def _detect_dims(df: pd.DataFrame) -> list[str]:
    """Return dimension column names: columns between DGUID (idx 2) and UOM."""
    cols = list(df.columns)
    if "UOM" not in cols:
        raise ValueError(f"Missing UOM column. Got: {cols[:12]}")
    return cols[3:cols.index("UOM")]


def _label_dims(dim_cols: list[str]) -> tuple[str | None, str | None]:
    """Identify which dimension is 'trade partner' and which is 'commodity'."""
    partner, commodity = None, None
    for c in dim_cols:
        lc = c.lower()
        if "partner" in lc:
            partner = c
        elif any(k in lc for k in ("import", "export", "commodity", "section", "chapter", "division", "product")):
            commodity = c

    # Positional fallback
    if partner is None and len(dim_cols) >= 1:
        partner = dim_cols[0]
    if commodity is None and len(dim_cols) >= 2:
        commodity = dim_cols[1]

    return partner, commodity


def _classify_row_type(
    partner_col: str | None,
    commodity_col: str | None,
    df: pd.DataFrame,
) -> pd.Series:
    def is_total(col: str | None) -> pd.Series:
        if col is None:
            return pd.Series(False, index=df.index)
        return df[col].str.contains(_TOTAL_RE, regex=True, na=False)

    pt = is_total(partner_col)
    ct = is_total(commodity_col)

    row_type = pd.Series("detail", index=df.index, dtype=object)
    row_type[~pt & ct]  = "country_total"
    row_type[pt  & ~ct] = "commodity_total"
    row_type[pt  & ct]  = "grand_total"
    return row_type


def _process(raw: pd.DataFrame, trade_type: str) -> pd.DataFrame:
    scale = _get_scale(raw)

    # National level only
    df = raw[raw["GEO"].str.strip() == _GEO_NATIONAL].copy()
    if df.empty:
        raise ValueError(f"No rows for GEO='{_GEO_NATIONAL}'. Available: {raw['GEO'].unique()[:5]}")

    dims = _detect_dims(df)
    partner_col, commodity_col = _label_dims(dims)
    LOG.info("     dims=%s  partner=%s  commodity=%s", dims, partner_col, commodity_col)

    row_type = _classify_row_type(partner_col, commodity_col, df)

    df["_v"] = pd.to_numeric(df["VALUE"], errors="coerce")
    df = df[df["_v"].notna() & (df["_v"] > 0)].copy()
    row_type = row_type.loc[df.index]

    result = pd.DataFrame({
        "date":       pd.to_datetime(df["REF_DATE"]).dt.to_period("M").astype(str),
        "trade_type": trade_type,
        "partner":    df[partner_col].str.strip() if partner_col else "Total",
        "commodity":  df[commodity_col].str.strip() if commodity_col else "Total",
        "value_cad":  (df["_v"] * scale).astype("int64"),
        "row_type":   row_type.values,
    })

    counts = result["row_type"].value_counts().to_dict()
    LOG.info("     %s: %d rows  %s", trade_type, len(result), counts)
    return result.reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(out: str = "data/canada_trade.parquet") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pid, trade_type in TABLES.items():
        raw_bytes = _download(pid)
        raw_df    = _open_csv(raw_bytes, pid)
        clean     = _process(raw_df, trade_type)
        frames.append(clean)

    df = pd.concat(frames, ignore_index=True)

    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    LOG.info("Saved → %s (%d rows)", out_path, len(df))

    # Quick validation summary (grand totals, last 6 months)
    gt = df[df["row_type"] == "grand_total"]
    if not gt.empty:
        summary = (
            gt.groupby(["date", "trade_type"])["value_cad"]
            .sum()
            .unstack("trade_type")
            .tail(6)
            .apply(lambda c: c / 1e9)
        )
        LOG.info("Recent grand totals (CAD billions):\n%s", summary.round(2).to_string())

    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download Statistics Canada trade tables")
    ap.add_argument("--out", default="data/canada_trade.parquet")
    run(out=ap.parse_args().out)

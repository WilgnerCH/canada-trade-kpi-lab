import pandas as pd
import os
import json
import re
from hs_lookup import get_hs_lookup, match_hs_description

# =========================
# CONFIG
# =========================

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    print("📥 Loading dataset...")
    df = pd.read_parquet(DATA_URL)
    print(f"✅ Dataset loaded: {len(df)} rows")
    return df


# =========================
# CLEAN (REMOVE DUPLICATION)
# =========================

def clean_data(df):
    print("🧹 Cleaning data (removing duplicates)...")

    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    print(f"✅ Clean dataset: {len(df_clean)} rows")
    return df_clean


# =========================
# MONTHLY SUMMARY
# =========================

def monthly_summary(df):

    monthly = (
        df.groupby(["date", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("📊 Monthly summary created")
    return monthly


# =========================
# COUNTRY SUMMARY
# =========================

def country_summary(df):

    country = (
        df.groupby(["Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("🌍 Country summary created")
    return country


# =========================
# PRODUCTS SUMMARY
# =========================

def product_summary(df):

    products = (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("📦 Product summary created")
    return products


# =========================
# 🧼 CLEAN PRODUCT NAME
# =========================

def clean_product_name(name):
    if not name:
        return name

    # Remove HS code no início (ex: 2709.00, 01.01, 0101 etc.)
    cleaned = re.sub(r'^\d+(\.\d+)*\s*', '', name)

    return cleaned.strip()


# =========================
# SAVE OUTPUTS
# =========================

def save_outputs(monthly, country, products):

    print("🔗 Loading HS lookup...")
    hs_lookup = get_hs_lookup()

    # -------------------------
    # MONTHLY JSON
    # -------------------------
    monthly_pivot = (
        monthly.pivot(index="date", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    monthly_json = [
        {
            "date": row["date"],
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "balance": float(
                row.get("Export", 0) - row.get("Import", 0)
            )
        }
        for _, row in monthly_pivot.iterrows()
    ]

    # -------------------------
    # COUNTRIES JSON
    # -------------------------
    countries_pivot = (
        country.pivot(index="Country", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    countries_json = [
        {
            "country": row["Country"],
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(row.get("Import", 0) + row.get("Export", 0))
        }
        for _, row in countries_pivot.iterrows()
    ]

    countries_json = sorted(countries_json, key=lambda x: x["total"], reverse=True)

    # -------------------------
    # PRODUCTS JSON (COM NOME LIMPO)
    # -------------------------
    products_pivot = (
        products.pivot(index="HS", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    products_json = []

    for _, row in products_pivot.iterrows():

        hs_code = row["HS"]

        # Nome original
        raw_name = match_hs_description(hs_code, hs_lookup)

        # Nome limpo
        name = clean_product_name(raw_name)

        products_json.append({
            "hs": str(hs_code),
            "name": name,
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(row.get("Import", 0) + row.get("Export", 0))
        })

    products_json = sorted(products_json, key=lambda x: x["total"], reverse=True)
    
    # Top 100 products for dashboards
    products_top100_json = products_json[:100]

    # -------------------------
    # SAVE FILES
    # -------------------------
    with open(f"{OUTPUT_DIR}/monthly.json", "w") as f:
        json.dump(monthly_json, f)
    
    with open(f"{OUTPUT_DIR}/countries.json", "w") as f:
        json.dump(countries_json, f)
    
    with open(f"{OUTPUT_DIR}/products.json", "w") as f:
        json.dump(products_json, f)
    
    with open(f"{OUTPUT_DIR}/products_top100.json", "w") as f:
        json.dump(products_top100_json, f)

    print("💾 JSON files saved in /data")


# =========================
# MAIN
# =========================

def main():

    df = load_data()

    df_clean = clean_data(df)

    m = monthly_summary(df_clean)
    c = country_summary(df_clean)
    p = product_summary(df_clean)

    save_outputs(m, c, p)

    print("🚀 Pipeline finished successfully!")


if __name__ == "__main__":
    main()

import pandas as pd
import os
import json

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
# SAVE OUTPUTS
# =========================

def save_outputs(monthly, country, products):

    # -------------------------
    # MONTHLY JSON (FOR LINE CHART)
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
            "exports": float(row.get("Export", 0))
        }
        for _, row in monthly_pivot.iterrows()
    ]

    # -------------------------
    # COUNTRIES JSON (BAR CHART)
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
    # PRODUCTS JSON (BAR CHART)
    # -------------------------
    products_pivot = (
        products.pivot(index="HS", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    products_json = [
        {
            "hs": str(row["HS"]),
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(row.get("Import", 0) + row.get("Export", 0))
        }
        for _, row in products_pivot.iterrows()
    ]

    products_json = sorted(products_json, key=lambda x: x["total"], reverse=True)

    # -------------------------
    # SAVE FILES
    # -------------------------
    with open(f"{OUTPUT_DIR}/monthly.json", "w") as f:
        json.dump(monthly_json, f)

    with open(f"{OUTPUT_DIR}/countries.json", "w") as f:
        json.dump(countries_json[:20], f)

    with open(f"{OUTPUT_DIR}/products.json", "w") as f:
        json.dump(products_json[:20], f)

    print("💾 JSON files saved in /data")


# =========================
# MAIN
# =========================

def main():

    df = load_data()

    # CLEAN DATA (ESSENCIAL)
    df_clean = clean_data(df)

    # BUILD AGGREGATIONS
    m = monthly_summary(df_clean)
    c = country_summary(df_clean)
    p = product_summary(df_clean)

    # SAVE FILES
    save_outputs(m, c, p)

    print("🚀 Pipeline finished successfully!")


if __name__ == "__main__":
    main()

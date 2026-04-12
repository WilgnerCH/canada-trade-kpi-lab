import pandas as pd
from huggingface_hub import HfApi
import os
import json

# =========================
# CONFIG
# =========================

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

OUTPUT_DIR = "data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    print("Loading dataset...")
    return pd.read_parquet(DATA_URL)


# =========================
# MONTHLY SUMMARY
# =========================

def monthly_summary(df):

    print("\n====================")
    print("DEBUG - TOTAL BRUTO 2026-02")
    print("====================")

    print("IMPORT:")
    print(
        df[(df["date"] == "2026-02") & (df["trade_type"] == "Import")]["Value"].sum()
    )

    print("EXPORT:")
    print(
        df[(df["date"] == "2026-02") & (df["trade_type"] == "Export")]["Value"].sum()
    )

    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    monthly = (
        df_clean.groupby(["date", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("\n====================")
    print("DEBUG - TOTAL FINAL 2026-02")
    print("====================")

    print("IMPORT:")
    print(
        monthly[
            (monthly["date"] == "2026-02") &
            (monthly["trade_type"] == "Import")
        ]["Value"].values
    )

    print("EXPORT:")
    print(
        monthly[
            (monthly["date"] == "2026-02") &
            (monthly["trade_type"] == "Export")
        ]["Value"].values
    )

    return monthly


# =========================
# COUNTRY SUMMARY
# =========================

def country_summary(df):
    return (
        df.groupby(["Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )


# =========================
# PRODUCTS SUMMARY
# =========================

def top_products(df):
    return (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )


# =========================
# SAVE OUTPUTS
# =========================

def save_outputs(monthly, country, products):

    # =========================
    # PARQUET (backend)
    # =========================
    monthly.to_parquet(f"{OUTPUT_DIR}/monthly.parquet", index=False)
    country.to_parquet(f"{OUTPUT_DIR}/country.parquet", index=False)
    products.to_parquet(f"{OUTPUT_DIR}/products.parquet", index=False)

    # =========================
    # MONTHLY JSON
    # =========================
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

    with open(f"{OUTPUT_DIR}/monthly.json", "w") as f:
        json.dump(monthly_json, f)

    # =========================
    # COUNTRIES JSON
    # =========================
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

    with open(f"{OUTPUT_DIR}/countries.json", "w") as f:
        json.dump(countries_json[:20], f)

    # =========================
    # PRODUCTS JSON
    # =========================
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

    with open(f"{OUTPUT_DIR}/products.json", "w") as f:
        json.dump(products_json[:20], f)

    print("✅ JSON + PARQUET files saved")


# =========================
# UPLOAD TO HUGGING FACE
# =========================

def upload_to_hf():
    api = HfApi()
    token = os.getenv("HF_TOKEN")

    for file in os.listdir(OUTPUT_DIR):
        api.upload_file(
            path_or_fileobj=f"{OUTPUT_DIR}/{file}",
            path_in_repo=file,
            repo_id="WilgnerCH/canada-trade-analytics",
            repo_type="dataset",
            token=token
        )

    print("✅ Upload complete")


# =========================
# MAIN PIPELINE
# =========================

def main():
    df = load_data()

    print("\n====================")
    print("DEBUG - DADOS 2026-02")
    print("====================")
    print(df[df["date"] == "2026-02"])

    print("\n====================")
    print("DEBUG - COUNTRIES DISPONÍVEIS")
    print("====================")
    print(df["Country"].unique())

    print("\n====================")
    print("TOP HS 2026-02")
    print("====================")

    print(
        df[df["date"] == "2026-02"]
        .groupby("HS")["Value"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
    )

    # =========================
    # PROCESSAMENTO
    # =========================
    m = monthly_summary(df)
    c = country_summary(df)
    p = top_products(df)

    save_outputs(m, c, p)
    upload_to_hf()

    print("🚀 Done")


if __name__ == "__main__":
    main()

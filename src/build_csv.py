import pandas as pd
import os
from hs_lookup import get_hs_lookup, match_hs_description

# =========================
# CONFIG
# =========================

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

CSV_OUTPUT_DIR = "data_csv"
os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATA
# =========================

def load_data():
    print("📥 Loading dataset...")
    df = pd.read_parquet(DATA_URL)
    print(f"✅ Dataset loaded: {len(df)} rows")
    return df


# =========================
# CLEAN DATA
# =========================

def clean_data(df):
    print("🧹 Cleaning data...")

    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    print(f"✅ Clean dataset: {len(df_clean)} rows")
    return df_clean


# =========================
# MONTHLY CSV
# =========================

def build_monthly_csv(df):

    monthly = (
        df.groupby(["date", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    monthly_pivot = (
        monthly.pivot(index="date", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    monthly_pivot.columns.name = None
    monthly_pivot.rename(columns={
        "Import": "imports",
        "Export": "exports"
    }, inplace=True)

    return monthly_pivot


# =========================
# COUNTRIES CSV
# =========================

def build_countries_csv(df):

    country = (
        df.groupby(["Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    pivot = (
        country.pivot(index="Country", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    pivot.columns.name = None

    pivot.rename(columns={
        "Country": "country",
        "Import": "imports",
        "Export": "exports"
    }, inplace=True)

    pivot["total"] = pivot["imports"] + pivot["exports"]

    return pivot.sort_values("total", ascending=False)


# =========================
# PRODUCTS CSV (COM NOME)
# =========================

def build_products_csv(df):

    print("🔗 Loading HS lookup...")
    hs_lookup = get_hs_lookup()

    products = (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    pivot = (
        products.pivot(index="HS", columns="trade_type", values="Value")
        .fillna(0)
        .reset_index()
    )

    pivot.columns.name = None

    pivot.rename(columns={
        "HS": "hs",
        "Import": "imports",
        "Export": "exports"
    }, inplace=True)

    pivot["total"] = pivot["imports"] + pivot["exports"]

    # 🔥 adicionar nome do produto
    pivot["name"] = pivot["hs"].apply(lambda x: match_hs_description(x, hs_lookup))

    # reorganizar colunas (profissional)
    pivot = pivot[["hs", "name", "imports", "exports", "total"]]

    return pivot.sort_values("total", ascending=False)


# =========================
# SAVE CSV
# =========================

def save_csv(monthly, countries, products):

    monthly.to_csv(f"{CSV_OUTPUT_DIR}/monthly.csv", index=False)
    countries.to_csv(f"{CSV_OUTPUT_DIR}/countries.csv", index=False)
    products.to_csv(f"{CSV_OUTPUT_DIR}/products.csv", index=False)

    print("💾 CSV files saved in /data_csv")


# =========================
# MAIN
# =========================

def main():

    df = load_data()
    df_clean = clean_data(df)

    monthly = build_monthly_csv(df_clean)
    countries = build_countries_csv(df_clean)
    products = build_products_csv(df_clean)

    save_csv(monthly, countries, products)

    print("🚀 CSV pipeline finished successfully!")


if __name__ == "__main__":
    main()

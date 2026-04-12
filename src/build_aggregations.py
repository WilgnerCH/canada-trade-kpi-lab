import pandas as pd
from huggingface_hub import HfApi
import os

DATA_URL = "https://huggingface.co/datasets/WilgnerCH/canada-trade-data/resolve/main/canada_trade_full.parquet"

OUTPUT_DIR = "data_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    print("Loading dataset...")
    return pd.read_parquet(DATA_URL)


def monthly_summary(df):

    # 🔴 DEBUG - ver valores antes de qualquer filtro
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

    # 🟡 AGREGAÇÃO CORRETA (remove duplicações estruturais)
    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    # 🔵 TOTAL FINAL
    monthly = (
        df_clean.groupby(["date", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    # 🟢 DEBUG FINAL
    print("\n====================")
    print("DEBUG - TOTAL FINAL 2026-02")
    print("====================")

    print("IMPORT:")
    print(
        monthly[(monthly["date"] == "2026-02") & (monthly["trade_type"] == "Import")]["Value"].values
    )

    print("EXPORT:")
    print(
        monthly[(monthly["date"] == "2026-02") & (monthly["trade_type"] == "Export")]["Value"].values
    )

    return monthly


def country_summary(df):
    return (
        df.groupby(["Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
        .sort_values("Value", ascending=False)
        .head(50)
    )


def top_products(df):
    return (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
        .sort_values("Value", ascending=False)
        .head(100)
    )


def save_outputs(monthly, country, products):
    monthly.to_parquet(f"{OUTPUT_DIR}/monthly.parquet", index=False)
    country.to_parquet(f"{OUTPUT_DIR}/country.parquet", index=False)
    products.to_parquet(f"{OUTPUT_DIR}/products.parquet", index=False)

    monthly.to_json(f"{OUTPUT_DIR}/monthly.json", orient="records")


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


def main():
    df = load_data()

    # =========================
    # DEBUG (TEMPORÁRIO)
    # =========================
    print("\n====================")
    print("DEBUG - DADOS 2026-02")
    print("====================")
    print(df[df["date"] == "2026-02"])

    print("\n====================")
    print("DEBUG - COUNTRIES DISPONÍVEIS")
    print("====================")
    print(df["Country"].unique())

    # =========================
    # PROCESSAMENTO
    # =========================
    m = monthly_summary(df)
    c = country_summary(df)
    p = top_products(df)

    save_outputs(m, c, p)
    upload_to_hf()

    print("Done 🚀")


if __name__ == "__main__":
    main()

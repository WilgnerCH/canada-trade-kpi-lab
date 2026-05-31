import pandas as pd
import os
import json
import re
from hs_lookup import get_hs_lookup, match_hs_description

# =========================
# HS2 NAMES
# =========================

HS2_NAMES = {
    "01": "Live animals",
    "02": "Meat and edible meat offal",
    "03": "Fish and seafood",
    "04": "Dairy products",
    "05": "Animal products n.e.s.",
    "06": "Live trees and plants",
    "07": "Vegetables",
    "08": "Fruit and nuts",
    "09": "Coffee, tea and spices",
    "10": "Cereals",
    "11": "Milling products",
    "12": "Oil seeds",
    "13": "Lac, gums and resins",
    "14": "Vegetable plaiting materials",
    "15": "Animal or vegetable fats and oils",
    "16": "Prepared meat and fish",
    "17": "Sugars and confectionery",
    "18": "Cocoa products",
    "19": "Prepared cereal products",
    "20": "Prepared vegetables and fruits",
    "21": "Miscellaneous food preparations",
    "22": "Beverages and spirits",
    "23": "Food industry residues",
    "24": "Tobacco products",
    "25": "Salt, sulphur, stone, cement",
    "26": "Ores, slag and ash",
    "27": "Mineral fuels and oils",
    "28": "Inorganic chemicals",
    "29": "Organic chemicals",
    "30": "Pharmaceutical products",
    "31": "Fertilizers",
    "32": "Tanning and dye extracts",
    "33": "Essential oils and cosmetics",
    "34": "Soap and cleaning products",
    "35": "Protein substances",
    "36": "Explosives",
    "37": "Photographic goods",
    "38": "Miscellaneous chemical products",
    "39": "Plastics",
    "40": "Rubber",
    "41": "Raw hides and skins",
    "42": "Leather goods",
    "43": "Furskins",
    "44": "Wood products",
    "45": "Cork products",
    "46": "Basketware",
    "47": "Pulp of wood",
    "48": "Paper and paperboard",
    "49": "Printed books and media",
    "50": "Silk",
    "51": "Wool",
    "52": "Cotton",
    "53": "Vegetable textile fibres",
    "54": "Man-made filaments",
    "55": "Man-made staple fibres",
    "56": "Wadding and felt",
    "57": "Carpets",
    "58": "Special woven fabrics",
    "59": "Impregnated textiles",
    "60": "Knitted fabrics",
    "61": "Knitted apparel",
    "62": "Woven apparel",
    "63": "Other textile articles",
    "64": "Footwear",
    "65": "Headgear",
    "66": "Umbrellas",
    "67": "Feathers and artificial flowers",
    "68": "Stone and cement articles",
    "69": "Ceramic products",
    "70": "Glass products",
    "71": "Precious stones and metals",
    "72": "Iron and steel",
    "73": "Articles of iron and steel",
    "74": "Copper",
    "75": "Nickel",
    "76": "Aluminium",
    "78": "Lead",
    "79": "Zinc",
    "80": "Tin",
    "81": "Other base metals",
    "82": "Tools and cutlery",
    "83": "Miscellaneous metal articles",
    "84": "Machinery and mechanical appliances",
    "85": "Electrical machinery",
    "86": "Railway equipment",
    "87": "Vehicles",
    "88": "Aircraft and spacecraft",
    "89": "Ships and boats",
    "90": "Optical and medical instruments",
    "91": "Clocks and watches",
    "92": "Musical instruments",
    "93": "Arms and ammunition",
    "94": "Furniture",
    "95": "Toys and sports equipment",
    "96": "Miscellaneous manufactured articles",
    "97": "Works of art"
}

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
    print("Loading dataset...")
    df = pd.read_parquet(DATA_URL)
    print(f"Dataset loaded: {len(df)} rows")
    return df


# =========================
# CLEAN (REMOVE DUPLICATION)
# =========================

def clean_data(df):
    print("Cleaning data (removing duplicates)...")

    df_clean = (
        df.groupby(["date", "trade_type", "Country", "HS"])["Value"]
        .sum()
        .reset_index()
    )

    print(f"Clean dataset: {len(df_clean)} rows")
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

    print("Monthly summary created")
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

    print("Country summary created")
    return country

# =========================
# COUNTRY MONTHLY SUMMARY
# =========================

def country_monthly_summary(df):

    country_monthly = (
        df.groupby(["date", "Country", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("Country monthly summary created")
    return country_monthly

# =========================
# PRODUCTS MONTHLY SUMMARY
# =========================

def products_monthly_summary(df):

    products_monthly = (
        df.groupby(["date", "HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("Products monthly summary created")
    return products_monthly

# =========================
# HS2 SUMMARY
# =========================

def hs2_summary(df):

    df = df.copy()

    df["hs2"] = (
        df["HS"]
        .astype(str)
        .str[:2]
    )

    hs2 = (
        df.groupby(
            ["hs2", "trade_type"]
        )["Value"]
        .sum()
        .reset_index()
    )

    print("HS2 summary created")
    return hs2

# =========================
# HS2 MONTHLY SUMMARY
# =========================

def hs2_monthly_summary(df):

    df = df.copy()

    df["hs2"] = (
        df["HS"]
        .astype(str)
        .str[:2]
    )

    hs2_monthly = (
        df.groupby(
            ["date", "hs2", "trade_type"]
        )["Value"]
        .sum()
        .reset_index()
    )

    print("HS2 monthly summary created")
    return hs2_monthly

# =========================
# PRODUCTS SUMMARY
# =========================

def product_summary(df):

    products = (
        df.groupby(["HS", "trade_type"])["Value"]
        .sum()
        .reset_index()
    )

    print("Product summary created")
    return products


# =========================
# CLEAN PRODUCT NAME
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

def save_outputs(
    monthly,
    country,
    products,
    country_monthly,
    products_monthly,
    hs2,
    hs2_monthly
):

    print("Loading HS lookup...")
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
    # COUNTRIES MONTHLY JSON
    # -------------------------

    country_monthly_pivot = (
        country_monthly
        .pivot_table(
            index=["date", "Country"],
            columns="trade_type",
            values="Value",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )

    countries_monthly_json = [
        {
            "date": row["date"],
            "country": row["Country"],
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(
                row.get("Import", 0) +
                row.get("Export", 0)
            )
        }
        for _, row in country_monthly_pivot.iterrows()
    ]

    countries_monthly_json = sorted(
        countries_monthly_json,
        key=lambda x: (x["date"], -x["total"])
    )    

    # -------------------------
    # PRODUCTS MONTHLY JSON
    # -------------------------
    
    products_monthly_pivot = (
        products_monthly
        .pivot_table(
            index=["date", "HS"],
            columns="trade_type",
            values="Value",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    
    products_monthly_json = []
    
    for _, row in products_monthly_pivot.iterrows():
    
        hs_code = row["HS"]
    
        raw_name = match_hs_description(
            hs_code,
            hs_lookup
        )
    
        name = clean_product_name(raw_name)
    
        products_monthly_json.append({
            "date": row["date"],
            "hs": str(hs_code),
            "name": name,
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(
                row.get("Import", 0)
                + row.get("Export", 0)
            )
        })
    
    print(
        f"Products monthly records: "
        f"{len(products_monthly_json)}"
    )
    
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
    # PRODUCTS MONTHLY TOP100
    # -------------------------
    
    top100_hs = set(
        product["hs"]
        for product in products_top100_json
    )
    
    products_monthly_top100_json = [
        row
        for row in products_monthly_json
        if row["hs"] in top100_hs
    ]
    
    print(
        f"Products monthly top100 records: "
        f"{len(products_monthly_top100_json)}"
    )

    # -------------------------
    # HS2 JSON
    # -------------------------
    
    hs2_pivot = (
        hs2.pivot(
            index="hs2",
            columns="trade_type",
            values="Value"
        )
        .fillna(0)
        .reset_index()
    )
    
    hs2_json = [
        {
            "hs2": row["hs2"],
            "hs2_name": HS2_NAMES.get(
                str(row["hs2"]).zfill(2),
                "Unknown"
            ),
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(
                row.get("Import", 0)
                + row.get("Export", 0)
            )
        }
        for _, row in hs2_pivot.iterrows()
    ]
    
    hs2_json = sorted(
        hs2_json,
        key=lambda x: x["total"],
        reverse=True
    )
    
    print(
        f"HS2 records: {len(hs2_json)}"
    )

    # -------------------------
    # HS2 MONTHLY JSON
    # -------------------------
    
    hs2_monthly_pivot = (
        hs2_monthly
        .pivot_table(
            index=["date", "hs2"],
            columns="trade_type",
            values="Value",
            aggfunc="sum"
        )
        .fillna(0)
        .reset_index()
    )
    
    hs2_monthly_json = [
        {
            "date": row["date"],
            "hs2": row["hs2"],
            "hs2_name": HS2_NAMES.get(
                str(row["hs2"]).zfill(2),
                "Unknown"
            ),
            "imports": float(row.get("Import", 0)),
            "exports": float(row.get("Export", 0)),
            "total": float(
                row.get("Import", 0)
                + row.get("Export", 0)
            )
        }
        for _, row in hs2_monthly_pivot.iterrows()
    ]
    
    # Ordena por mês e depois pelo maior total
    hs2_monthly_json = sorted(
        hs2_monthly_json,
        key=lambda x: (
            x["date"],
            -x["total"]
        )
    )
    
    print(
        f"HS2 monthly records: "
        f"{len(hs2_monthly_json)}"
    )
    
    # -------------------------
    # SAVE FILES
    # -------------------------
    with open(f"{OUTPUT_DIR}/monthly.json", "w") as f:
        json.dump(monthly_json, f)
    
    with open(f"{OUTPUT_DIR}/countries.json", "w") as f:
        json.dump(countries_json, f)

    with open(f"{OUTPUT_DIR}/countries_monthly.json", "w") as f:
        json.dump(countries_monthly_json, f)
    
    with open(f"{OUTPUT_DIR}/products.json", "w") as f:
        json.dump(products_json, f)
    
    with open(f"{OUTPUT_DIR}/products_top100.json", "w") as f:
        json.dump(products_top100_json, f)

    with open(
        f"{OUTPUT_DIR}/products_monthly.json",
        "w"
    ) as f:
        json.dump(products_monthly_json, f)

    with open(
        f"{OUTPUT_DIR}/products_monthly_top100.json",
        "w"
    ) as f:
        json.dump(products_monthly_top100_json, f)

    with open(
        f"{OUTPUT_DIR}/hs2_summary.json",
        "w"
    ) as f:
        json.dump(hs2_json, f)

    with open(
        f"{OUTPUT_DIR}/hs2_monthly.json",
        "w"
    ) as f:
        json.dump(hs2_monthly_json, f)
    
    print("JSON files saved in /data")

# =========================
# MAIN
# =========================

def main():

    df = load_data()

    df_clean = clean_data(df)

    m = monthly_summary(df_clean)
    c = country_summary(df_clean)
    p = product_summary(df_clean)
    cm = country_monthly_summary(df_clean)
    pm = products_monthly_summary(df_clean)
    h2 = hs2_summary(df_clean)
    h2m = hs2_monthly_summary(df_clean)
    
    save_outputs(
        m,
        c,
        p,
        cm,
        pm,
        h2,
        h2m
    )

    print("Pipeline finished successfully!")


if __name__ == "__main__":
    main()

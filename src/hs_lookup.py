import pandas as pd


# =========================
# NORMALIZE HS
# =========================
def normalize_hs(hs_code):
    """
    Converte:
    2709.00.10 -> 270900
    """
    if pd.isna(hs_code):
        return ""

    hs = str(hs_code).replace(".", "").replace(" ", "")

    return hs[:6]


# =========================
# LOAD LOOKUP
# =========================
def get_hs_lookup():

    df = pd.read_csv("data/hs6_lookup.csv")

    # garantir formato correto
    df["hs6"] = df["hs6"].astype(str).str.strip().str.zfill(6)
    df["description"] = df["description"].astype(str).str.strip()

    lookup = dict(zip(df["hs6"], df["description"]))

    print(f"📘 HS lookup loaded: {len(lookup)} entries")

    return lookup


# =========================
# MATCH FUNCTION
# =========================
def match_hs_description(hs_code, lookup):

    hs6 = normalize_hs(hs_code)

    # tenta match progressivo (robusto)
    return (
        lookup.get(hs6) or
        lookup.get(hs6[:4]) or
        lookup.get(hs6[:2]) or
        f"HS {hs6}"
    )

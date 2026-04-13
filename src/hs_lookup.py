import pandas as pd

def get_hs_lookup():

    df = pd.read_csv("data/hs6_lookup.csv")

    # garantir string
    df["hs6"] = df["hs6"].astype(str).str.zfill(6)

    # criar dicionário
    lookup = dict(zip(df["hs6"], df["description"]))

    return lookup


def match_hs_description(hs_code, lookup):

    # limpar formato (8703.23.00 → 87032300)
    hs_code = str(hs_code).replace(".", "").replace(" ", "")

    # tentar match progressivo
    return (
        lookup.get(hs_code[:6]) or
        lookup.get(hs_code[:4]) or
        lookup.get(hs_code[:2]) or
        hs_code
    )

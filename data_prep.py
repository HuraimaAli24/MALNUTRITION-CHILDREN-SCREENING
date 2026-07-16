"""
DHS Children's Recode (KR) data loader and cleaner
for the multi-task malnutrition risk model.

Works on both DHS Model Datasets and real country KR files
(India/Afghanistan) since variable structure is identical.
"""

import pandas as pd
import pyreadstat
import numpy as np

# DHS stores z-scores as integers scaled by 100 (e.g. -134 = -1.34 SD)
# Flag codes: 9998 = flagged/biologically implausible, 9999 = missing
Z_FLAG_VALUES = {9998, 9999, -9998, -9999}

DHS_COLUMNS = {
    "hw1": "age_months",
    "hw2": "weight_kg_raw",       # already *10 in DHS, needs /10
    "hw3": "height_cm_raw",       # already *10 in DHS, needs /10
    "hw70": "haz_raw",            # height-for-age z-score (stunting), *100
    "hw71": "waz_raw",            # weight-for-age z-score (underweight), *100
    "hw72": "whz_raw",            # weight-for-height z-score (wasting), *100
    "b4": "sex_code",             # 1 = male, 2 = female
    "v190": "wealth_index",       # 1 (poorest) - 5 (richest)
    "v024": "region",
    "v025": "residence_type",     # 1 = urban, 2 = rural
    "v106": "mother_education",   # 0=none,1=primary,2=secondary,3=higher
    "v012": "mother_age",
}


def load_dhs_kr(dta_path: str) -> pd.DataFrame:
    """Load a DHS Children's Recode .DTA file and return a raw dataframe
    with just the columns we need, renamed to readable names."""
    df, meta = pyreadstat.read_dta(dta_path)
    cols_present = [c for c in DHS_COLUMNS if c in df.columns]
    sub = df[cols_present].rename(columns=DHS_COLUMNS)
    return sub


def clean_and_engineer(sub: pd.DataFrame) -> pd.DataFrame:
    """Clean raw DHS fields, rescale z-scores, drop flagged/missing rows,
    and engineer model-ready features."""
    df = sub.copy()

    # --- Rescale z-scores (DHS stores them x100) ---
    for zcol in ["haz_raw", "waz_raw", "whz_raw"]:
        df[zcol] = df[zcol].apply(
            lambda v: np.nan if pd.isna(v) or v in Z_FLAG_VALUES or abs(v) > 600 else v / 100.0
        )

    df = df.rename(columns={"haz_raw": "haz", "waz_raw": "waz", "whz_raw": "whz"})

    # --- Rescale weight/height (DHS stores x10) ---
    df["weight_kg"] = df["weight_kg_raw"].apply(lambda v: np.nan if pd.isna(v) or v >= 9990 else v / 10.0)
    df["height_cm"] = df["height_cm_raw"].apply(lambda v: np.nan if pd.isna(v) or v >= 9990 else v / 10.0)
    df = df.drop(columns=["weight_kg_raw", "height_cm_raw"])

    # --- Drop rows without valid anthropometry (not measured / fully flagged) ---
    df = df.dropna(subset=["haz", "waz", "whz", "age_months"])

    # --- Sex encoding: 0 = male, 1 = female ---
    df["sex"] = df["sex_code"].map({1: 0, 2: 1})
    df = df.drop(columns=["sex_code"])

    # --- WHO severity categories (used for classification head + reporting) ---
    def severity(z):
        if z < -3:
            return 2  # severe
        elif z < -2:
            return 1  # moderate
        else:
            return 0  # normal

    df["stunting_class"] = df["haz"].apply(severity)
    df["underweight_class"] = df["waz"].apply(severity)
    df["wasting_class"] = df["whz"].apply(severity)

    # --- Clip age to plausible under-5 range ---
    df = df[(df["age_months"] >= 0) & (df["age_months"] <= 59)]

    # --- Fill remaining minor missingness in covariates ---
    df["mother_education"] = df["mother_education"].fillna(df["mother_education"].median())
    df["wealth_index"] = df["wealth_index"].fillna(df["wealth_index"].median())

    df = df.reset_index(drop=True)
    return df


def get_feature_target_arrays(df: pd.DataFrame):
    """Split cleaned dataframe into feature matrix and multi-task targets."""
    feature_cols = [
        "age_months", "sex", "wealth_index", "region",
        "residence_type", "mother_education", "mother_age",
    ]
    X = df[feature_cols].copy()

    targets = {
        "haz": df["haz"].values.astype("float32"),
        "waz": df["waz"].values.astype("float32"),
        "whz": df["whz"].values.astype("float32"),
        "stunting_class": df["stunting_class"].values.astype("int64"),
        "underweight_class": df["underweight_class"].values.astype("int64"),
        "wasting_class": df["wasting_class"].values.astype("int64"),
    }
    return X, targets, feature_cols


if __name__ == "__main__":
    raw = load_dhs_kr("data/zzkr62dt/ZZKR62FL.DTA")
    print("Raw shape:", raw.shape)

    clean = clean_and_engineer(raw)
    print("Clean shape:", clean.shape)
    print()
    print(clean[["age_months", "haz", "waz", "whz",
                  "stunting_class", "underweight_class", "wasting_class"]].describe())
    print()
    print("Severity distribution (stunting):")
    print(clean["stunting_class"].value_counts())

    clean.to_csv("data/clean_kr.csv", index=False)
    print("\nSaved cleaned data to data/clean_kr.csv")

"""
processor.py
Handles all data loading and cleaning.
"""

import os
import pandas as pd
from thefuzz import process

CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")

# Minimum fuzzy match score (0-100) to accept a result
FUZZY_THRESHOLD = 60

# Known irritants for sensitive skin
RED_FLAGS = [
    "fragrance",
    "alcohol denat",
    "sodium lauryl sulfate",
    "sodium laureth sulfate",
    "methylparaben",
    "propylparaben",
    "butylparaben",
    "ethylparaben",
    "formaldehyde",
    "phthalate",
    "oxybenzone",
    "triclosan",
]


def load_data() -> pd.DataFrame:
    """Load and clean the cosmetics CSV dataset."""
    try:
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")  # utf-8-sig strips BOM character
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {CSV_PATH}. "
            "Make sure cosmetic_p.csv is in the same folder as this file."
        )

    # Clean up
    df["ingredients"] = df["ingredients"].fillna("")
    df["name"] = df["name"].fillna("Unknown")
    df["brand"] = df["brand"].fillna("Unknown")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0)

    return df


def parse_ingredients(ingredients_str: str) -> set:
    """Parse a comma-separated ingredients string into a lowercase set."""
    if not isinstance(ingredients_str, str):
        return set()
    return {i.strip().lower() for i in ingredients_str.split(",") if i.strip()}


def find_product(df: pd.DataFrame, product_name: str) -> pd.Series | None:
    """
    Look up a product by name using a three-step strategy:
    1. Exact match (fastest)
    2. Partial string match
    3. Fuzzy match via thefuzz (handles typos and accent variations)

    Returns the first matching row as a Series, or None if not found.
    """
    name_lower = product_name.lower()

    # Step 1: Exact match
    match = df[df["name"].str.lower() == name_lower]
    if not match.empty:
        return match.iloc[0]

    # Step 2: Partial match
    match = df[df["name"].str.lower().str.contains(name_lower, na=False)]
    if not match.empty:
        return match.iloc[0]

    # Step 3: Fuzzy match — handles typos, missing accents, etc.
    all_names = df["name"].tolist()
    result = process.extractOne(product_name, all_names, score_cutoff=FUZZY_THRESHOLD)
    if result:
        best_match_name, score = result[0], result[1]
        match = df[df["name"] == best_match_name]
        if not match.empty:
            return match.iloc[0]

    return None

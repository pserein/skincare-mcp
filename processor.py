"""
processor.py
Handles all data loading and cleaning.
"""

import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")

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
    Look up a product by name. Tries exact match first, then partial match.
    Returns the first matching row as a Series, or None if not found.
    """
    name_lower = product_name.lower()

    # Exact match
    match = df[df["name"].str.lower() == name_lower]
    if not match.empty:
        return match.iloc[0]

    # Partial match
    match = df[df["name"].str.lower().str.contains(name_lower, na=False)]
    if not match.empty:
        return match.iloc[0]

    return None

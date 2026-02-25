"""
database.py
Cached SQLite query functions for the Skincare MCP app.
"""

import os
import sqlite3
import pandas as pd
import streamlit as st
from thefuzz import process

DB_PATH = os.path.join(os.path.dirname(__file__), "skincare.db")
FUZZY_THRESHOLD = 60


def get_connection():
    """Create a new SQLite connection. Called fresh per query (thread-safe)."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_all_products() -> pd.DataFrame:
    """Load the full products table into memory once and cache it."""
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM products", conn)
    conn.close()
    return df


@st.cache_data
def get_all_product_names() -> list[str]:
    """Return sorted list of all product names for autocomplete."""
    conn = get_connection()
    cursor = conn.execute("SELECT name FROM products ORDER BY name ASC;")
    names = [row[0] for row in cursor.fetchall()]
    conn.close()
    return names


@st.cache_data
def search_product_by_name(product_name: str) -> pd.Series | None:
    """
    Look up a product by exact name first, then fuzzy match.
    Returns a single DataFrame row as a Series, or None if not found.
    """
    conn = get_connection()

    # 1. Exact match (case-insensitive)
    df = pd.read_sql(
        "SELECT * FROM products WHERE LOWER(name) = LOWER(?);",
        conn,
        params=(product_name,),
    )
    if not df.empty:
        conn.close()
        return df.iloc[0]

    # 2. Partial match
    df = pd.read_sql(
        "SELECT * FROM products WHERE LOWER(name) LIKE LOWER(?);",
        conn,
        params=(f"%{product_name}%",),
    )
    if not df.empty:
        conn.close()
        return df.iloc[0]

    conn.close()

    # 3. Fuzzy match against all names
    all_names = get_all_product_names()
    result = process.extractOne(product_name, all_names, score_cutoff=FUZZY_THRESHOLD)
    if result:
        return search_product_by_name(result[0])

    return None


@st.cache_data
def get_product_by_exact_name(name: str) -> pd.Series | None:
    """Fetch a product row by exact name. Used after selectbox selection."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM products WHERE name = ?;",
        conn,
        params=(name,),
    )
    conn.close()
    return df.iloc[0] if not df.empty else None


@st.cache_data
def get_top_rated_products(min_rank: float = 4.5, n: int = 8) -> pd.DataFrame:
    """Fetch top-rated products for the 'no results' suggestion fallback."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT name, brand, Label, rank FROM products WHERE rank >= ? ORDER BY RANDOM() LIMIT ?;",
        conn,
        params=(min_rank, n),
    )
    conn.close()
    return df.rename(columns={"name": "Product", "brand": "Brand", "Label": "Category", "rank": "Rating"})

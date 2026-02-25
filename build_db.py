"""
build_db.py
Run once locally to create skincare.db from cosmetic_p.csv.

Usage:
    python build_db.py
"""

import os
import sqlite3
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")
DB_PATH = os.path.join(os.path.dirname(__file__), "skincare.db")


def build():
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    # Clean
    df["ingredients"] = df["ingredients"].fillna("")
    df["name"] = df["name"].fillna("Unknown")
    df["brand"] = df["brand"].fillna("Unknown")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0)
    for col in ["Combination", "Dry", "Normal", "Oily", "Sensitive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    print(f"Writing {len(df):,} rows to {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)

    df.to_sql("products", conn, if_exists="replace", index=True, index_label="id")

    # Indexes for fast lookups
    conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON products(name);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_brand ON products(brand);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON products(Label);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_rank ON products(rank);")

    conn.commit()
    conn.close()
    print("Done! skincare.db is ready.")


if __name__ == "__main__":
    build()

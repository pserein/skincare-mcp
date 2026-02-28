"""
mcp_server.py
MCP server interface — exposes tools to Claude.
Keeps all business logic in engine.py and processor.py.
"""

import os
import sqlite3
import requests
from mcp.server.fastmcp import FastMCP
from processor import load_data
from engine import get_similar_products, get_red_flags

mcp = FastMCP("skincare-recommender")

DB_PATH = os.path.join(os.path.dirname(__file__), "skincare.db")
OPEN_BEAUTY_API = "https://world.openbeautyfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "SkincareMCP/1.0 (educational project)"}

# Load data once at startup
df = load_data()


# ── Existing tools ─────────────────────────────────────────────────────────────

@mcp.tool()
def find_similar_products(product_name: str) -> str:
    """Find skincare products with similar ingredients using Jaccard similarity"""
    if not isinstance(product_name, str) or not product_name.strip():
        return "Please provide a valid product name."
    return get_similar_products(df, product_name.strip())


@mcp.tool()
def check_red_flags(product_name: str) -> str:
    """Check if a product contains known irritants for sensitive skin"""
    if not isinstance(product_name, str) or not product_name.strip():
        return "Please provide a valid product name."
    return get_red_flags(df, product_name.strip())


# ── New tool: SQL query ────────────────────────────────────────────────────────

@mcp.tool()
def query_database(sql: str) -> str:
    """
    Run a read-only SQL SELECT query against the local skincare SQLite database.
    The database has a 'products' table with columns:
    name, brand, Label (category), price, rank, ingredients,
    Combination, Dry, Normal, Oily, Sensitive (skin type flags, 0 or 1).
    Only SELECT statements are allowed.
    Example: SELECT name, brand, price FROM products WHERE Label = 'Moisturizer' AND rank >= 4.5 LIMIT 5
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return "Only SELECT queries are allowed for safety."

    if not os.path.exists(DB_PATH):
        return "Database not found. Run build_db.py first."

    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No results found for that query."

        # Format as readable table
        headers = rows[0].keys()
        lines = [" | ".join(headers)]
        lines.append("-" * len(lines[0]))
        for row in rows[:20]:  # cap at 20 rows
            lines.append(" | ".join(str(row[h]) for h in headers))

        if len(rows) > 20:
            lines.append(f"... and {len(rows) - 20} more rows.")

        return "\n".join(lines)

    except sqlite3.Error as e:
        return f"SQL error: {str(e)}"


# ── New tool: Live API search ──────────────────────────────────────────────────

@mcp.tool()
def search_live_products(query: str, max_results: int = 5) -> str:
    """
    Search Open Beauty Facts for real-time beauty product data.
    Useful for products not in the local database (e.g. CeraVe, La Roche-Posay).
    Returns product name, brand, and ingredients for each result.
    Limited to beauty and personal care products only.
    """
    if not isinstance(query, str) or not query.strip():
        return "Please provide a valid search query."

    max_results = min(max_results, 10)

    try:
        params = {
            "search_terms": query.strip(),
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": max_results,
            "fields": "product_name,brands,ingredients_text,url",
        }
        response = requests.get(OPEN_BEAUTY_API, params=params, headers=HEADERS, timeout=10)
        response.raise_for_status()
        data = response.json()

        products = data.get("products", [])
        if not products:
            return f"No live results found for '{query}'."

        lines = [f"Live results for '{query}' from Open Beauty Facts:\n"]
        for p in products:
            name = p.get("product_name", "").strip()
            brand = p.get("brands", "Unknown").strip()
            ingredients = p.get("ingredients_text", "Not available").strip()
            url = p.get("url", "")

            if not name:
                continue

            ing_preview = ingredients[:300] + "..." if len(ingredients) > 300 else ingredients
            lines.append(f"**{name}** by {brand}")
            lines.append(f"Ingredients: {ing_preview}")
            if url:
                lines.append(f"More info: {url}")
            lines.append("")

        return "\n".join(lines)

    except requests.exceptions.Timeout:
        return "Live API request timed out. Try again."
    except requests.exceptions.RequestException as e:
        return f"Live API error: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
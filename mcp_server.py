"""
mcp_server.py
MCP server interface — exposes tools to Claude.
Keeps all business logic in engine.py and processor.py.
"""

from mcp.server.fastmcp import FastMCP
from processor import load_data
from engine import get_similar_products, get_red_flags

mcp = FastMCP("skincare-recommender")

# Load data once at startup
df = load_data()


@mcp.tool()
def find_similar_products(product_name: str) -> str:
    """Find skincare products with similar ingredients"""
    if not isinstance(product_name, str) or not product_name.strip():
        return "Please provide a valid product name."
    return get_similar_products(df, product_name.strip())


@mcp.tool()
def check_red_flags(product_name: str) -> str:
    """Check if a product contains known irritants for sensitive skin"""
    if not isinstance(product_name, str) or not product_name.strip():
        return "Please provide a valid product name."
    return get_red_flags(df, product_name.strip())


if __name__ == "__main__":
    mcp.run(transport="stdio")

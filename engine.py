"""
engine.py
Core recommendation logic — ingredient similarity and red flag detection.
This is the "brain" of the recommender, kept separate from the MCP interface.
"""

import pandas as pd
from processor import RED_FLAGS, find_product, parse_ingredients


def get_similar_products(df: pd.DataFrame, product_name: str) -> str:
    """
    Find products with the most ingredient overlap using Jaccard Similarity.
    Returns a formatted string result.
    """
    product = find_product(df, product_name)
    if product is None:
        return f"Could not find a product named '{product_name}' in the database."

    source_ingredients = parse_ingredients(product["ingredients"])
    if not source_ingredients:
        return f"No ingredient data available for '{product['name']}'."

    results = []
    for _, row in df.iterrows():
        if row["name"] == product["name"]:
            continue

        other_ingredients = parse_ingredients(row["ingredients"])
        if not other_ingredients:
            continue

        # Jaccard Similarity: intersection / union
        intersection = len(source_ingredients & other_ingredients)
        union = len(source_ingredients | other_ingredients)
        similarity = intersection / union

        if similarity > 0.1:  # at least 10% overlap
            results.append({
                "name": row["name"],
                "brand": row["brand"],
                "price": row["price"],
                "rank": row["rank"],
                "similarity": round(similarity * 100, 1),
                "shared_ingredients": intersection,
            })

    if not results:
        return f"No similar products found for '{product['name']}'."

    results.sort(key=lambda x: x["similarity"], reverse=True)
    top = results[:5]

    lines = [f"Top {len(top)} products similar to **{product['name']}** by {product['brand']}:\n"]
    for r in top:
        lines.append(
            f"- {r['name']} by {r['brand']} | ${r['price']} | "
            f"Rating: {r['rank']} | {r['similarity']}% ingredient match "
            f"({r['shared_ingredients']} shared ingredients)"
        )
    return "\n".join(lines)


def get_red_flags(df: pd.DataFrame, product_name: str) -> str:
    """
    Check a product for known irritants.
    Returns a formatted string result.
    """
    product = find_product(df, product_name)
    if product is None:
        return f"Could not find a product named '{product_name}' in the database."

    ingredients_lower = product["ingredients"].lower()

    found_flags = [
        flag.title()
        for flag in RED_FLAGS
        if flag in ingredients_lower
    ]

    if not found_flags:
        return (
            f"✅ No common irritants found in **{product['name']}** by {product['brand']}. "
            f"Looks good for sensitive skin!"
        )

    flags_str = ", ".join(found_flags)
    return (
        f"⚠️ **{product['name']}** by {product['brand']} contains potential irritants:\n"
        f"{flags_str}\n\n"
        f"These ingredients may cause reactions in sensitive skin types."
    )

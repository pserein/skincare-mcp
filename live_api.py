"""
live_api.py
Live product data from Open Beauty Facts API.
No API key required.
"""

import requests
import streamlit as st

BASE_URL = "https://world.openbeautyfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "SkincareMCP/1.0 (educational project)"}


@st.cache_data(ttl=3600)  # Cache for 1 hour
def search_live_products(query: str, page_size: int = 6) -> list[dict]:
    """
    Search Open Beauty Facts for products matching the query.
    Returns a list of cleaned product dicts.
    TTL of 1 hour so repeated searches don't hammer the API.
    """
    try:
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": page_size,
            "fields": "product_name,brands,ingredients_text,image_front_url,url,categories_tags",
        }
        response = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=8)
        response.raise_for_status()
        data = response.json()

        products = []
        for p in data.get("products", []):
            name = p.get("product_name", "").strip()
            brand = p.get("brands", "").strip()
            ingredients = p.get("ingredients_text", "").strip()
            image_url = p.get("image_front_url", "")
            product_url = p.get("url", "")

            # Skip entries with no name
            if not name:
                continue

            products.append({
                "name": name,
                "brand": brand if brand else "Unknown",
                "ingredients": ingredients if ingredients else "Not available",
                "image_url": image_url,
                "url": product_url,
            })

        return products

    except requests.exceptions.Timeout:
        return []
    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []


@st.cache_data(ttl=3600)
def get_live_product_detail(product_name: str, brand: str) -> dict | None:
    """
    Fetch a specific product by name and brand.
    Used when user clicks into a live result.
    """
    query = f"{product_name} {brand}".strip()
    results = search_live_products(query, page_size=3)
    for r in results:
        if r["name"].lower() == product_name.lower():
            return r
    return results[0] if results else None

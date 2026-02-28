"""
live_api.py
Live product data from Open Beauty Facts API.
No API key required.
"""

import requests
import streamlit as st

BASE_URL = "https://world.openbeautyfacts.org/cgi/search.pl"
HEADERS = {"User-Agent": "SkincareMCP/1.0 (educational project - contact: skincaremcp@gmail.com)"}


@st.cache_data(ttl=3600)
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
            "fields": "product_name,brands,ingredients_text,image_front_url,url",
        }
        response = requests.get(
            BASE_URL,
            params=params,
            headers=HEADERS,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        products = []
        for p in data.get("products", []):
            name = p.get("product_name", "").strip()
            brand = p.get("brands", "").strip()
            ingredients = p.get("ingredients_text", "").strip()
            image_url = p.get("image_front_url", "")
            product_url = p.get("url", "")

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
        st.warning("Live API timed out. Try again in a moment.")
        return []
    except requests.exceptions.ConnectionError:
        st.warning("Could not connect to Open Beauty Facts. Check your connection.")
        return []
    except Exception as e:
        st.warning(f"Live API error: {str(e)}")
        return []

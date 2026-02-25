"""
app.py
Streamlit dashboard for the Skincare MCP recommendation engine.
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from database import (
    load_all_products,
    get_all_product_names,
    search_product_by_name,
    get_product_by_exact_name,
    get_top_rated_products,
)
from live_api import search_live_products
from build_db import build as build_database

# ── Auto-build DB if missing (e.g. on Streamlit Cloud first boot) ─────────────
if not os.path.exists(os.path.join(os.path.dirname(__file__), "skincare.db")):
    build_database()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Ingredient Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rhode-inspired styling ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #2C2C2C !important;
    }
    .stApp {
        background-color: #F5F0EB;
        color: #2C2C2C;
    }
    [data-testid="stSidebar"] {
        background-color: #EDE8E3;
        border-right: 1px solid #D9D3CC;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #4A4038 !important;
    }
    [data-testid="stSidebar"] h1 {
        color: #2C2C2C !important;
        font-weight: 500 !important;
        font-size: 16px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    h1 {
        color: #2C2C2C !important;
        font-weight: 400 !important;
        letter-spacing: 0.05em !important;
        text-transform: uppercase !important;
        font-size: 28px !important;
        margin-bottom: 4px !important;
    }
    h2, h3, h4 {
        color: #2C2C2C !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em !important;
    }
    [data-testid="metric-container"] {
        background-color: #EDE8E3;
        border: 1px solid #D9D3CC;
        border-radius: 8px;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label {
        color: #8A8480 !important;
        font-size: 11px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #2C2C2C !important;
        font-size: 16px !important;
        font-weight: 500 !important;
    }
    [data-baseweb="input"] input {
        background-color: #EDE8E3 !important;
        color: #2C2C2C !important;
    }
    [data-baseweb="base-input"] {
        border: 1px solid #C9C3BC !important;
        border-radius: 6px !important;
    }
    .stButton button {
        background-color: #2C2C2C !important;
        color: #F5F0EB !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 12px 20px !important;
        font-weight: 500 !important;
    }
    .stButton button p, .stButton button span {
        color: #F5F0EB !important;
    }
    .stButton button:hover {
        background-color: #444 !important;
    }
    [data-baseweb="select"] div {
        background-color: #EDE8E3 !important;
        color: #2C2C2C !important;
    }
    .stSuccess {
        background-color: #E8F0E8 !important;
        border: 1px solid #B8D4B8 !important;
        border-radius: 6px !important;
        color: #2C4A2C !important;
    }
    .stWarning {
        background-color: #F5EDDF !important;
        border: 1px solid #E0CEAA !important;
        border-radius: 6px !important;
        color: #5C4A2A !important;
    }
    .block-container {
        padding-top: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        max-width: 1200px !important;
    }
    hr {
        border-color: #D9D3CC !important;
        margin: 1.5rem 0 !important;
    }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] p,
    [data-testid="stRadio"] span {
        color: #2C2C2C !important;
        font-size: 13px !important;
        letter-spacing: 0.03em !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "user_history.csv")
CHART_COLOR = "#8A7D6E"
CHART_SCALE = ["#C9BFB5", "#A89A8C", "#8A7D6E", "#6B6056", "#4A4038", "#2C2C2C", "#1A1A1A"]

RED_FLAGS = [
    "fragrance", "alcohol denat", "sodium lauryl sulfate",
    "sodium laureth sulfate", "methylparaben", "propylparaben",
    "butylparaben", "ethylparaben", "formaldehyde", "phthalate",
    "oxybenzone", "triclosan",
]
SKIN_TYPE_COLS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_history():
    if os.path.exists(HISTORY_PATH):
        return pd.read_csv(HISTORY_PATH)
    return None

@st.cache_resource
def build_tfidf(_df):
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[a-zA-Z][a-zA-Z0-9\-\(\)\.]*",
        ngram_range=(1, 2),
        min_df=2,
    )
    matrix = vectorizer.fit_transform(_df["ingredients"].tolist())
    return vectorizer, matrix


# ── Cached computation helpers ────────────────────────────────────────────────

@st.cache_data
def get_red_flags(ingredients: str) -> list[str]:
    ingredients_lower = ingredients.lower()
    return [f.title() for f in RED_FLAGS if f in ingredients_lower]


@st.cache_data
def get_top_ingredients(_tfidf_matrix, feature_names: list, idx: int, top_n: int = 12) -> list[tuple]:
    scores = _tfidf_matrix[idx].toarray().flatten()
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [(str(feature_names[i]), round(float(scores[i]), 4)) for i in top_idx if scores[i] > 0]


@st.cache_data
def get_similar_products(_tfidf_matrix, idx: int, top_n: int = 6) -> list[dict]:
    df = load_all_products()
    scores = cosine_similarity(_tfidf_matrix[idx], _tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[::-1]
    results = []
    product_name = df.iloc[idx]["name"]
    for i in top_indices:
        if df.iloc[i]["name"] == product_name:
            continue
        score = scores[i]
        if score < 0.1 or len(results) == top_n:
            break
        row = df.iloc[i]
        results.append({
            "Product": row["name"],
            "Brand": row["brand"],
            "Price": f"${row['price']}",
            "Rating": star_rating(row["rank"]),
            "Match": f"{round(score * 100, 1)}%",
        })
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def star_rating(score):
    full = int(score)
    return "★" * full + "☆" * (5 - full)


def show_product(df, vectorizer, tfidf_matrix, product):
    st.markdown("---")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Product", product["name"])
    col2.metric("Brand", product["brand"])
    col3.metric("Category", product["Label"])
    col4.metric("Price", f"${product['price']}")
    col5.metric("Rating", f"{star_rating(product['rank'])}  {product['rank']}")

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        st.markdown("#### Skin Type Compatibility")
        cols = st.columns(5)
        for i, skin in enumerate(SKIN_TYPE_COLS):
            val = int(product.get(skin, 0))
            cols[i].metric(skin, "✓" if val == 1 else "—")

    with right:
        st.markdown("#### Irritant Check")
        found_flags = get_red_flags(product["ingredients"])
        if found_flags:
            st.warning(f"**Potential irritants:** {', '.join(found_flags)}")
        else:
            st.success("No common irritants found — suitable for sensitive skin.")

    st.markdown("<br>", unsafe_allow_html=True)

    left2, right2 = st.columns(2)
    idx = df[df["name"] == product["name"]].index[0]
    feature_names = vectorizer.get_feature_names_out()

    with left2:
        st.markdown("#### Top Ingredients by TF-IDF Weight")
        top_ingredients = get_top_ingredients(tfidf_matrix, list(feature_names), idx)
        if top_ingredients:
            ing_df = pd.DataFrame(top_ingredients, columns=["Ingredient", "TF-IDF Score"])
            fig = px.bar(
                ing_df, x="TF-IDF Score", y="Ingredient", orientation="h",
                color="TF-IDF Score", color_continuous_scale=CHART_SCALE,
            )
            fig.update_layout(
                paper_bgcolor="#F5F0EB", plot_bgcolor="#F5F0EB",
                yaxis=dict(autorange="reversed"), showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                font=dict(color="#2C2C2C", size=12, family="Inter"),
                coloraxis_showscale=False,
            )
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#2C2C2C"))
            fig.update_yaxes(showgrid=False, tickfont=dict(color="#2C2C2C"))
            st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.markdown("#### Similar Products")
        results = get_similar_products(tfidf_matrix, idx)
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True, height=280)
        else:
            st.info("No similar products found.")


# ── Load everything ───────────────────────────────────────────────────────────
df = load_all_products()          # ← now from SQLite, not CSV
history_df = load_history()
vectorizer, tfidf_matrix = build_tfidf(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Skincare Engine")
st.sidebar.markdown("TF-IDF · Cosine Similarity · MCP · SQLite")
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["Product Search", "Browse Products", "Data Explorer", "User History"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df):,}** products · **{df['brand'].nunique()}** brands")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Product Search  (now uses SQLite lookups)
# ══════════════════════════════════════════════════════════════════════════════
if page == "Product Search":
    st.title("Product Search")
    st.markdown("Search any product to find ingredient-based alternatives and check for irritants.")
    st.markdown("<br>", unsafe_allow_html=True)

    all_product_names = [""] + get_all_product_names()

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.selectbox(
            "",
            options=all_product_names,
            index=0,
            placeholder="e.g. Crème de la Mer, Vitamin C serum, moisturizer...",
            label_visibility="collapsed",
        )
    with col_btn:
        if st.button("Random"):
            random_pick = df["name"].sample(1).iloc[0]
            st.session_state["query"] = random_pick

    if "query" in st.session_state and not query:
        query = st.session_state["query"]

    if query:
        # ── SQLite lookup ──
        product = search_product_by_name(query)
        if product is None:
            st.error(f"No product found matching '{query}' in local database.")
            st.markdown("**Try one of these popular products:**")
            st.dataframe(get_top_rated_products(), use_container_width=True, hide_index=True)
        else:
            show_product(df, vectorizer, tfidf_matrix, product)

        # ── Live results from Open Beauty Facts ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🌐 Live Results from Open Beauty Facts", expanded=False):
            with st.spinner("Fetching live data..."):
                live_results = search_live_products(query)

            if not live_results:
                st.info("No live results found or API unavailable.")
            else:
                st.markdown(f"Found **{len(live_results)}** live results for **{query}**")
                st.markdown("<br>", unsafe_allow_html=True)
                for i, p in enumerate(live_results):
                    col_img, col_info = st.columns([1, 4])
                    with col_img:
                        if p["image_url"]:
                            st.image(p["image_url"], width=80)
                        else:
                            st.markdown("🧴")
                    with col_info:
                        st.markdown(f"**{p['name']}** — {p['brand']}")
                        if p["ingredients"] != "Not available":
                            # Show first 200 chars of ingredients
                            ing_preview = p["ingredients"][:200]
                            if len(p["ingredients"]) > 200:
                                ing_preview += "..."
                            st.caption(f"Ingredients: {ing_preview}")
                        else:
                            st.caption("Ingredients: Not available")
                        if p["url"]:
                            st.markdown(f"[View on Open Beauty Facts]({p['url']})")
                    if i < len(live_results) - 1:
                        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Browse Products
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Browse Products":
    st.title("Browse Products")
    st.markdown(f"Filter and explore all **{len(df):,}** products.")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_cat = st.selectbox("Category", ["All"] + sorted(df["Label"].unique().tolist()))
    with col2:
        selected_brand = st.selectbox("Brand", ["All"] + sorted(df["brand"].unique().tolist()))
    with col3:
        skin_filter = st.selectbox("Skin Type", ["All"] + SKIN_TYPE_COLS)
    with col4:
        min_rating = st.selectbox("Min Rating", [0, 3.0, 3.5, 4.0, 4.5], index=0)

    filtered = df.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["Label"] == selected_cat]
    if selected_brand != "All":
        filtered = filtered[filtered["brand"] == selected_brand]
    if skin_filter != "All":
        filtered = filtered[filtered[skin_filter] == 1]
    if min_rating:
        filtered = filtered[filtered["rank"] >= min_rating]

    st.markdown(f"**{len(filtered):,} products match your filters.**")

    display_df = filtered[["name", "brand", "Label", "price", "rank"]].rename(
        columns={"name": "Product", "brand": "Brand", "Label": "Category", "price": "Price ($)", "rank": "Rating"}
    ).sort_values("Rating", ascending=False)
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400, column_config={
        "Product": st.column_config.TextColumn(width="large"),
        "Brand": st.column_config.TextColumn(width="medium"),
        "Category": st.column_config.TextColumn(width="medium"),
        "Price ($)": st.column_config.NumberColumn(width="small", format="$%.0f"),
        "Rating": st.column_config.NumberColumn(width="small", format="%.1f ⭐"),
    })

    st.markdown("---")
    st.markdown("#### Inspect a Product")
    if not filtered.empty:
        selected_name = st.selectbox("Select a product", filtered["name"].tolist())
        if selected_name:
            product = get_product_by_exact_name(selected_name)
            if product is not None:
                show_product(df, vectorizer, tfidf_matrix, product)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown(f"**{len(df):,}** products · **{df['brand'].nunique()}** brands · **{df['Label'].nunique()}** categories")
    st.markdown("<br>", unsafe_allow_html=True)

    def style_chart(fig):
        fig.update_layout(
            paper_bgcolor="#F5F0EB", plot_bgcolor="#F5F0EB",
            font=dict(color="#2C2C2C", size=13, family="Inter"),
            title_font=dict(color="#2C2C2C", size=14, family="Inter"),
            margin=dict(l=10, r=10, t=50, b=10),
            legend=dict(font=dict(color="#2C2C2C")),
        )
        fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
        fig.update_yaxes(showgrid=True, gridcolor="#D9D3CC", zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
        return fig

    col1, col2 = st.columns(2)
    with col1:
        cat_counts = df["Label"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig1 = px.bar(cat_counts, x="Category", y="Count", title="Products by Category", color="Count", color_continuous_scale=CHART_SCALE)
        st.plotly_chart(style_chart(fig1), use_container_width=True)
    with col2:
        fig2 = px.histogram(df[df["rank"] > 0], x="rank", nbins=20, title="Rating Distribution", color_discrete_sequence=[CHART_COLOR])
        fig2.update_layout(xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(style_chart(fig2), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(df[df["price"] > 0], x="Label", y="price", title="Price by Category", color="Label",
                      color_discrete_sequence=CHART_SCALE)
        fig3.update_layout(showlegend=False, xaxis_title="", yaxis_title="Price ($)")
        st.plotly_chart(style_chart(fig3), use_container_width=True)
    with col4:
        top_brands = df["brand"].value_counts().head(15).reset_index()
        top_brands.columns = ["Brand", "Count"]
        fig4 = px.bar(top_brands, x="Count", y="Brand", orientation="h", title="Top 15 Brands",
                      color="Count", color_continuous_scale=CHART_SCALE)
        fig4.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_chart(fig4), use_container_width=True)

    skin_counts = {col: int(df[col].sum()) for col in SKIN_TYPE_COLS}
    fig5 = px.pie(values=list(skin_counts.values()), names=list(skin_counts.keys()),
                  title="Products Per Skin Type", color_discrete_sequence=CHART_SCALE)
    fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#2C2C2C"))
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: User History
# ══════════════════════════════════════════════════════════════════════════════
elif page == "User History":
    st.title("User History")
    st.markdown("Synthetic interaction logs for Offline Reinforcement Learning — State → Action → Reward.")
    st.markdown("<br>", unsafe_allow_html=True)

    if history_df is None:
        st.warning("user_history.csv not found. Run `python generate_user_history.py` first.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(history_df):,}")
        c2.metric("Simulated Users", history_df["user_id"].nunique())
        c3.metric("Avg Reward", round(history_df["reward"].mean(), 3))

        st.markdown("<br>", unsafe_allow_html=True)

        def style_chart(fig):
            fig.update_layout(
                paper_bgcolor="#F5F0EB", plot_bgcolor="#F5F0EB",
                font=dict(color="#2C2C2C", size=13, family="Inter"),
                title_font=dict(color="#2C2C2C", size=14, family="Inter"),
                margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(font=dict(color="#2C2C2C")),
            )
            fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
            fig.update_yaxes(showgrid=True, gridcolor="#D9D3CC", zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
            return fig

        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(history_df, x="reward", nbins=30, title="Reward Distribution",
                                color_discrete_sequence=[CHART_COLOR])
            st.plotly_chart(style_chart(fig1), use_container_width=True)
        with col2:
            concern_over_time = history_df.groupby("timestep")[["dryness", "acne", "sensitivity", "oiliness"]].mean().reset_index()
            fig2 = px.line(concern_over_time, x="timestep", y=["dryness", "acne", "sensitivity", "oiliness"],
                           title="Avg Skin Concerns Over Time", color_discrete_sequence=["#A89A8C", "#8A7D6E", "#6B6056", "#4A4038"])
            fig2.update_layout(legend_title="Concern")
            st.plotly_chart(style_chart(fig2), use_container_width=True)

        avg_reward = history_df.groupby("label")["reward"].mean().sort_values(ascending=False).reset_index()
        avg_reward.columns = ["Category", "Avg Reward"]
        fig3 = px.bar(avg_reward, x="Category", y="Avg Reward", title="Avg Reward by Product Category",
                      color="Avg Reward", color_continuous_scale=CHART_SCALE)
        st.plotly_chart(style_chart(fig3), use_container_width=True)

        st.markdown("#### Sample Interaction Log")
        st.dataframe(history_df.head(20), use_container_width=True, hide_index=True)

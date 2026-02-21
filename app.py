"""
app.py
Streamlit dashboard for the Skincare MCP recommendation engine.

Run with:
    streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skincare Ingredient Engine",
    page_icon="🧴",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "user_history.csv")
FUZZY_THRESHOLD = 60

RED_FLAGS = [
    "fragrance", "alcohol denat", "sodium lauryl sulfate",
    "sodium laureth sulfate", "methylparaben", "propylparaben",
    "butylparaben", "ethylparaben", "formaldehyde", "phthalate",
    "oxybenzone", "triclosan",
]

SKIN_TYPE_COLS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    df["ingredients"] = df["ingredients"].fillna("")
    df["name"] = df["name"].fillna("Unknown")
    df["brand"] = df["brand"].fillna("Unknown")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0)
    return df

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

def find_product(df, product_name):
    match = df[df["name"].str.lower() == product_name.lower()]
    if not match.empty:
        return match.iloc[0]
    match = df[df["name"].str.lower().str.contains(product_name.lower(), na=False)]
    if not match.empty:
        return match.iloc[0]
    result = process.extractOne(product_name, df["name"].tolist(), score_cutoff=FUZZY_THRESHOLD)
    if result:
        match = df[df["name"] == result[0]]
        if not match.empty:
            return match.iloc[0]
    return None

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
history_df = load_history()
vectorizer, tfidf_matrix = build_tfidf(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("Skincare Ingredient Engine")
st.sidebar.markdown("Built with TF-IDF, cosine similarity, and MCP.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Product Search", "Data Explorer", "User History"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Product Search
# ══════════════════════════════════════════════════════════════════════════════
if page == "Product Search":
    st.title("Product Search")
    st.markdown("Search any product to find similar alternatives and check for irritants.")

    query = st.text_input("Enter a product name", placeholder="e.g. Crème de la Mer")

    if query:
        product = find_product(df, query)

        if product is None:
            st.error(f"No product found matching '{query}'. Try a different name.")
        else:
            # ── Product header ─────────────────────────────────────────────
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Product", product["name"])
            col2.metric("Brand", product["brand"])
            col3.metric("Rating", f"{product['rank']} / 5.0")

            col4, col5 = st.columns(2)
            col4.metric("Category", product["Label"])
            col5.metric("Price", f"${product['price']}")

            # ── Skin type compatibility ────────────────────────────────────
            st.markdown("#### Skin Type Compatibility")
            skin_cols = st.columns(5)
            for i, skin in enumerate(SKIN_TYPE_COLS):
                val = int(product.get(skin, 0))
                skin_cols[i].metric(skin, "✓" if val == 1 else "✗")

            # ── Red flag check ─────────────────────────────────────────────
            st.markdown("#### Irritant Check")
            ingredients_lower = product["ingredients"].lower()
            found_flags = [f.title() for f in RED_FLAGS if f in ingredients_lower]

            if found_flags:
                st.warning(f"**Potential irritants found:** {', '.join(found_flags)}")
            else:
                st.success("No common irritants found. Suitable for sensitive skin.")

            # ── Top ingredients chart ──────────────────────────────────────
            st.markdown("#### Top Ingredients by TF-IDF Weight")
            idx = df[df["name"] == product["name"]].index[0]
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix[idx].toarray().flatten()
            top_n = 15
            top_idx = np.argsort(tfidf_scores)[::-1][:top_n]
            top_ingredients = [(feature_names[i], tfidf_scores[i]) for i in top_idx if tfidf_scores[i] > 0]

            if top_ingredients:
                ing_df = pd.DataFrame(top_ingredients, columns=["Ingredient", "TF-IDF Score"])
                fig = px.bar(
                    ing_df,
                    x="TF-IDF Score",
                    y="Ingredient",
                    orientation="h",
                    color="TF-IDF Score",
                    color_continuous_scale="Blues",
                    title="Most Distinctive Ingredients (higher = more unique to this product)",
                )
                fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # ── Similar products ───────────────────────────────────────────
            st.markdown("#### Similar Products")
            idx = df[df["name"] == product["name"]].index[0]
            scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            top_indices = np.argsort(scores)[::-1]

            results = []
            for i in top_indices:
                if df.iloc[i]["name"] == product["name"]:
                    continue
                score = scores[i]
                if score < 0.1 or len(results) == 8:
                    break
                row = df.iloc[i]
                results.append({
                    "Product": row["name"],
                    "Brand": row["brand"],
                    "Category": row["Label"],
                    "Price": f"${row['price']}",
                    "Rating": row["rank"],
                    "Similarity": f"{round(score * 100, 1)}%",
                })

            if results:
                st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
            else:
                st.info("No similar products found.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown(f"Exploring **{len(df):,}** products across **{df['Label'].nunique()}** categories.")

    col1, col2 = st.columns(2)

    # Products by category
    with col1:
        cat_counts = df["Label"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig1 = px.bar(cat_counts, x="Category", y="Count", title="Products by Category", color="Count", color_continuous_scale="Blues")
        st.plotly_chart(fig1, use_container_width=True)

    # Rating distribution
    with col2:
        fig2 = px.histogram(df[df["rank"] > 0], x="rank", nbins=20, title="Product Rating Distribution", color_discrete_sequence=["#4A90D9"])
        fig2.update_layout(xaxis_title="Rating", yaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # Price distribution by category
    with col3:
        fig3 = px.box(df[df["price"] > 0], x="Label", y="price", title="Price Distribution by Category", color="Label")
        fig3.update_layout(showlegend=False, xaxis_title="Category", yaxis_title="Price ($)")
        st.plotly_chart(fig3, use_container_width=True)

    # Top brands by product count
    with col4:
        top_brands = df["brand"].value_counts().head(15).reset_index()
        top_brands.columns = ["Brand", "Count"]
        fig4 = px.bar(top_brands, x="Count", y="Brand", orientation="h", title="Top 15 Brands by Product Count", color="Count", color_continuous_scale="Blues")
        fig4.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)

    # Skin type coverage
    st.markdown("#### Skin Type Coverage Across Dataset")
    skin_counts = {col: int(df[col].sum()) for col in SKIN_TYPE_COLS}
    fig5 = px.pie(
        values=list(skin_counts.values()),
        names=list(skin_counts.keys()),
        title="Products Suitable Per Skin Type",
    )
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: User History (RL Dataset)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "User History":
    st.title("User History — Offline RL Dataset")
    st.markdown("Synthetic interaction logs generated for Offline Reinforcement Learning (State → Action → Reward).")

    if history_df is None:
        st.warning("user_history.csv not found. Run `python generate_user_history.py` first.")
    else:
        st.markdown(f"**{len(history_df):,} interaction records** across **{history_df['user_id'].nunique()}** simulated users.")

        col1, col2 = st.columns(2)

        # Reward distribution
        with col1:
            fig1 = px.histogram(history_df, x="reward", nbins=30, title="Reward Distribution", color_discrete_sequence=["#4A90D9"])
            fig1.update_layout(xaxis_title="Reward", yaxis_title="Count")
            st.plotly_chart(fig1, use_container_width=True)

        # Average skin concerns over timesteps
        with col2:
            concern_over_time = history_df.groupby("timestep")[["dryness", "acne", "sensitivity", "oiliness"]].mean().reset_index()
            fig2 = px.line(concern_over_time, x="timestep", y=["dryness", "acne", "sensitivity", "oiliness"], title="Average Skin Concerns Over Time")
            fig2.update_layout(xaxis_title="Timestep", yaxis_title="Concern Level (0-1)", legend_title="Concern")
            st.plotly_chart(fig2, use_container_width=True)

        # Average reward per product category
        avg_reward = history_df.groupby("label")["reward"].mean().sort_values(ascending=False).reset_index()
        avg_reward.columns = ["Category", "Avg Reward"]
        fig3 = px.bar(avg_reward, x="Category", y="Avg Reward", title="Average Reward by Product Category", color="Avg Reward", color_continuous_scale="RdYlGn")
        st.plotly_chart(fig3, use_container_width=True)

        # Sample of the raw data
        st.markdown("#### Sample Interaction Log")
        st.dataframe(history_df.head(20), use_container_width=True, hide_index=True)


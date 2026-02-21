import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from thefuzz import process

st.set_page_config(
    page_title="Skincare Ingredient Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #2C2C2C !important;
    }

    * {
        color: inherit;
    }

    p, span, label, div, h1, h2, h3, h4, h5, li, td, th {
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

    [data-testid="stSidebar"] .stMarkdown p {
        color: #4A4038;
        font-size: 13px;
        letter-spacing: 0.03em;
    }

    [data-testid="stSidebar"] h1 {
        color: #2C2C2C;
        font-weight: 500;
        font-size: 16px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
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

    .stTextInput input {
        background-color: #EDE8E3 !important;
        border: 1px solid #C9C3BC !important;
        border-radius: 6px !important;
        color: #2C2C2C !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
    }

    .stTextInput input:focus {
        border-color: #8A8480 !important;
        box-shadow: none !important;
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

    .stButton button:hover {
        background-color: #444 !important;
    }

    .stSelectbox select, [data-testid="stSelectbox"] {
        background-color: #EDE8E3 !important;
        border: 1px solid #C9C3BC !important;
        border-radius: 6px !important;
        color: #2C2C2C !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #D9D3CC !important;
        border-radius: 8px !important;
        background-color: #F5F0EB !important;
    }

    [data-testid="stDataFrame"] * {
        color: #2C2C2C !important;
        background-color: #F5F0EB !important;
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
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px !important;
    }

    hr {
        border-color: #D9D3CC !important;
        margin: 1.5rem 0 !important;
    }

    .stRadio label, .stRadio span, [data-testid="stRadio"] label,
    [data-testid="stRadio"] p, [data-testid="stRadio"] span {
        color: #2C2C2C !important;
        font-size: 13px !important;
        letter-spacing: 0.03em !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {
        color: #2C2C2C !important;
    }
</style>
""", unsafe_allow_html=True)

CSV_PATH = os.path.join(os.path.dirname(__file__), "cosmetic_p.csv")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "user_history.csv")
FUZZY_THRESHOLD = 60
CHART_COLOR = "#8A7D6E"
CHART_SCALE = ["#C9BFB5", "#A89A8C", "#8A7D6E", "#6B6056", "#4A4038", "#2C2C2C", "#1A1A1A"]

RED_FLAGS = [
    "fragrance", "alcohol denat", "sodium lauryl sulfate",
    "sodium laureth sulfate", "methylparaben", "propylparaben",
    "butylparaben", "ethylparaben", "formaldehyde", "phthalate",
    "oxybenzone", "triclosan",
]
SKIN_TYPE_COLS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]

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

def star_rating(score):
    full = int(score)
    return "★" * full + "☆" * (5 - full)

def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="#F5F0EB",
        plot_bgcolor="#F5F0EB",
        font=dict(color="#2C2C2C", size=13, family="Inter"),
        title_font=dict(color="#2C2C2C", size=14, family="Inter"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(font=dict(color="#2C2C2C")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
    fig.update_yaxes(showgrid=True, gridcolor="#D9D3CC", zeroline=False, tickfont=dict(color="#2C2C2C"), title_font=dict(color="#2C2C2C"))
    return fig

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
        ingredients_lower = product["ingredients"].lower()
        found_flags = [f.title() for f in RED_FLAGS if f in ingredients_lower]
        if found_flags:
            st.warning(f"**Potential irritants:** {', '.join(found_flags)}")
        else:
            st.success("No common irritants found.")

    st.markdown("<br>", unsafe_allow_html=True)

    left2, right2 = st.columns(2)

    with left2:
        st.markdown("#### Top Ingredients by TF-IDF Weight")
        idx = df[df["name"] == product["name"]].index[0]
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        top_idx = np.argsort(tfidf_scores)[::-1][:12]
        top_ingredients = [(feature_names[i], round(float(tfidf_scores[i]), 4)) for i in top_idx if tfidf_scores[i] > 0]

        if top_ingredients:
            ing_df = pd.DataFrame(top_ingredients, columns=["Ingredient", "TF-IDF Score"])
            fig = px.bar(
                ing_df, x="TF-IDF Score", y="Ingredient", orientation="h",
                color="TF-IDF Score", color_continuous_scale=CHART_SCALE,
            )
            fig = style_chart(fig)
            fig.update_layout(
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
            )
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    with right2:
        st.markdown("#### Similar Products")
        idx = df[df["name"] == product["name"]].index[0]
        scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        top_indices = np.argsort(scores)[::-1]
        results = []
        for i in top_indices:
            if df.iloc[i]["name"] == product["name"]:
                continue
            score = scores[i]
            if score < 0.1 or len(results) == 6:
                break
            row = df.iloc[i]
            results.append({
                "Product": row["name"],
                "Brand": row["brand"],
                "Price": f"${row['price']}",
                "Rating": star_rating(row["rank"]),
                "Match": f"{round(score * 100, 1)}%",
            })
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True, height=280)
        else:
            st.info("No similar products found.")

df = load_data()
history_df = load_history()
vectorizer, tfidf_matrix = build_tfidf(df)

st.sidebar.title("Skincare Engine")
st.sidebar.markdown("TF-IDF · Cosine Similarity · MCP")
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["Product Search", "Browse Products", "Data Explorer", "User History"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**{len(df):,}** products · **{df['brand'].nunique()}** brands")

if page == "Product Search":
    st.title("Product Search")
    st.markdown("Search any product to find ingredient-based alternatives and check for irritants.")
    st.markdown("<br>", unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input("", placeholder="e.g. Crème de la Mer, Vitamin C serum, moisturizer...", label_visibility="collapsed")
    with col_btn:
        if st.button("Random"):
            query = df["name"].sample(1).iloc[0]
            st.session_state["query"] = query

    if "query" in st.session_state and not query:
        query = st.session_state["query"]

    if query:
        product = find_product(df, query)
        if product is None:
            st.error(f"No product found matching '{query}'.")
            st.markdown("**Try one of these popular products:**")
            suggestions = df[df["rank"] >= 4.5][["name", "brand", "Label", "rank"]].sample(8).rename(
                columns={"name": "Product", "brand": "Brand", "Label": "Category", "rank": "Rating"}
            )
            st.dataframe(suggestions, use_container_width=True, hide_index=True)
        else:
            show_product(df, vectorizer, tfidf_matrix, product)

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
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=400)

    st.markdown("---")
    st.markdown("#### Inspect a Product")
    if not filtered.empty:
        selected_name = st.selectbox("Select a product", filtered["name"].tolist())
        if selected_name:
            product = df[df["name"] == selected_name].iloc[0]
            show_product(df, vectorizer, tfidf_matrix, product)

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.markdown(f"**{len(df):,}** products · **{df['brand'].nunique()}** brands · **{df['Label'].nunique()}** categories")
    st.markdown("<br>", unsafe_allow_html=True)

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
        fig4 = style_chart(fig4)
        fig4.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    skin_counts = {col: int(df[col].sum()) for col in SKIN_TYPE_COLS}
    skin_df = pd.DataFrame(list(skin_counts.items()), columns=["Skin Type", "Count"]).sort_values("Count", ascending=True)
    fig5 = px.bar(skin_df, x="Count", y="Skin Type", orientation="h", title="Products Per Skin Type",
                  color="Count", color_continuous_scale=CHART_SCALE)
    fig5 = style_chart(fig5)
    fig5.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
    st.plotly_chart(fig5, use_container_width=True)

elif page == "User History":
    st.title("User History")
    st.markdown("Synthetic interaction logs for Offline Reinforcement Learning.")
    st.markdown("<br>", unsafe_allow_html=True)

    if history_df is None:
        st.warning("user_history.csv not found.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", f"{len(history_df):,}")
        c2.metric("Simulated Users", history_df["user_id"].nunique())
        c3.metric("Avg Reward", round(history_df["reward"].mean(), 3))

        st.markdown("<br>", unsafe_allow_html=True)

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
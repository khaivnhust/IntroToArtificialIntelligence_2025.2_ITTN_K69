"""
app.py -- Streamlit demo for the H&M Hybrid Recommendation System.

This file contains ONLY the UI layer.  All model inference logic is
delegated to ``src.models.InferencePipeline``.

Run with:   streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TOP_K_RECOMMENDATIONS  # noqa: E402
from src.models.inference_pipeline import InferencePipeline  # noqa: E402
from src.preprocessing.data_loader import DataLoaderPolars  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="H&M Fashion Recommender",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═══════════════════════════════════════════════════════════════════════════
# Custom CSS (no emoji, professional dark theme)
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* -- Global typography ------------------------------------------------ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* -- Header banner ---------------------------------------------------- */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #e94560;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #a8b2d1;
        font-size: 1.05rem;
        margin: 0;
    }

    /* -- Stat cards ------------------------------------------------------- */
    .stats-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    .stat-card {
        flex: 1;
        background: linear-gradient(145deg, #1e2a3a, #253345);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e94560;
    }
    .stat-card .stat-label {
        font-size: 0.82rem;
        color: #8892b0;
        margin-top: 0.2rem;
    }

    /* -- Product card grid ------------------------------------------------ */
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1.2rem;
    }
    .product-card {
        background: linear-gradient(160deg, #1b2838, #1e3048);
        border: 1px solid rgba(233, 69, 96, 0.12);
        border-radius: 14px;
        padding: 1.2rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .product-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(233, 69, 96, 0.2);
    }
    .product-rank {
        display: inline-block;
        background: linear-gradient(135deg, #e94560, #c73e54);
        color: #fff;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .product-name {
        color: #ccd6f6;
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 0.5rem;
    }
    .product-meta {
        font-size: 0.82rem;
        color: #8892b0;
        line-height: 1.55;
    }
    .product-score {
        margin-top: 0.7rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #64ffda;
    }

    /* -- User profile card ------------------------------------------------ */
    .user-profile {
        background: linear-gradient(145deg, #1e2a3a, #253345);
        border: 1px solid rgba(233, 69, 96, 0.15);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .user-profile h3 {
        color: #e94560;
        margin: 0 0 0.8rem 0;
        font-size: 1.1rem;
    }
    .profile-row {
        display: flex;
        justify-content: space-between;
        padding: 0.35rem 0;
        border-bottom: 1px solid rgba(136, 146, 176, 0.1);
    }
    .profile-key { color: #8892b0; font-size: 0.88rem; }
    .profile-val { color: #ccd6f6; font-size: 0.88rem; font-weight: 500; }

    /* -- Image placeholder ------------------------------------------------ */
    .img-placeholder {
        width: 100%;
        aspect-ratio: 3 / 4;
        background: linear-gradient(135deg, #2a3a50, #1e2a3a);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #4a5a6a;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Cached data / model loading
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading datasets ...")
def load_datasets():
    """Load all Parquet datasets and return the loader plus DataFrames."""
    data_loader = DataLoaderPolars()
    train_df, test_df, customers_df, articles_df = data_loader.load_all_dataframes()
    return data_loader, train_df, test_df, customers_df, articles_df


@st.cache_resource(show_spinner="Loading model and building pipeline ...")
def build_inference_pipeline(_train_df: pl.DataFrame) -> InferencePipeline:
    """Construct the inference pipeline (model + baseline + feature extractor)."""
    return InferencePipeline(train_df=_train_df)


# ═══════════════════════════════════════════════════════════════════════════
# Metadata lookup helpers
# ═══════════════════════════════════════════════════════════════════════════
def lookup_article_metadata(articles_df: pl.DataFrame, item_id: int) -> dict:
    """Return a dict of article attributes for *item_id*, or {}."""
    if "item_id" in articles_df.columns:
        row = articles_df.filter(pl.col("item_id") == item_id)
    elif "article_id" in articles_df.columns:
        row = articles_df.filter(pl.col("article_id") == item_id)
    else:
        return {}
    if row.height == 0:
        return {}
    return row.row(0, named=True)


def lookup_customer_profile(customers_df: pl.DataFrame, user_id: int) -> dict:
    """Return a dict of customer attributes for *user_id*, or {}."""
    row = customers_df.filter(pl.col("user_id") == user_id)
    if row.height == 0:
        return {}
    return row.row(0, named=True)


# ═══════════════════════════════════════════════════════════════════════════
# UI rendering helpers
# ═══════════════════════════════════════════════════════════════════════════
def render_customer_profile(customer_info: dict, user_id: int) -> None:
    """Render the customer profile card as HTML."""
    age = customer_info.get("age", "N/A")
    club_status = customer_info.get("club_member_status", "N/A")
    is_fn_member = "Yes" if customer_info.get("FN", 0) == 1.0 else "No"
    is_active = "Yes" if customer_info.get("Active", 0) == 1.0 else "No"
    fashion_news = customer_info.get("fashion_news_frequency", "N/A")

    st.markdown(f"""
    <div class="user-profile">
        <h3>Customer Profile — user_id: {user_id}</h3>
        <div class="profile-row"><span class="profile-key">Age</span><span class="profile-val">{age}</span></div>
        <div class="profile-row"><span class="profile-key">Club Status</span><span class="profile-val">{club_status}</span></div>
        <div class="profile-row"><span class="profile-key">FN Member</span><span class="profile-val">{is_fn_member}</span></div>
        <div class="profile-row"><span class="profile-key">Active</span><span class="profile-val">{is_active}</span></div>
        <div class="profile-row"><span class="profile-key">Fashion News</span><span class="profile-val">{fashion_news}</span></div>
    </div>
    """, unsafe_allow_html=True)


def render_product_cards(recommendations, articles_df: pl.DataFrame) -> None:
    """Render the product card grid as HTML."""
    cards_html = '<div class="product-grid">'

    for rank, (item_id, score) in enumerate(recommendations, 1):
        article_info = lookup_article_metadata(articles_df, item_id)

        product_name = article_info.get("prod_name", f"Item #{item_id}")
        product_type = article_info.get("product_type_name", "—")
        colour = article_info.get("colour_group_name", "—")
        department = article_info.get("department_name", "—")
        group = article_info.get("product_group_name", "—")
        section = article_info.get("section_name", "—")
        description = article_info.get("detail_desc", "")

        if description and len(str(description)) > 80:
            description = str(description)[:80] + "..."

        score_html = (
            f'<div class="product-score">Score: {score:.4f}</div>' if score > 0 else ""
        )

        cards_html += f"""
        <div class="product-card">
            <div class="img-placeholder">No Image</div>
            <span class="product-rank">#{rank}</span>
            <div class="product-name">{product_name}</div>
            <div class="product-meta">
                <b>Type:</b> {product_type}<br>
                <b>Colour:</b> {colour}<br>
                <b>Department:</b> {department}<br>
                <b>Group:</b> {group}<br>
                <b>Section:</b> {section}
            </div>
            {score_html}
        </div>
        """

    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def render_details_table(recommendations, articles_df: pl.DataFrame) -> None:
    """Render an expandable details table below the product cards."""
    with st.expander("Full details table"):
        table_rows = []
        for rank, (item_id, score) in enumerate(recommendations, 1):
            article_info = lookup_article_metadata(articles_df, item_id)
            table_rows.append({
                "Rank": rank,
                "Item ID": item_id,
                "Name": article_info.get("prod_name", "—"),
                "Type": article_info.get("product_type_name", "—"),
                "Colour": article_info.get("colour_group_name", "—"),
                "Department": article_info.get("department_name", "—"),
                "Group": article_info.get("product_group_name", "—"),
                "Score": f"{score:.4f}" if score > 0 else "N/A",
            })
        st.dataframe(
            pl.DataFrame(table_rows),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Main application
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>H&M Fashion Recommender</h1>
        <p>Hybrid NCF + Visual Features — Personalized Top-12 Product Recommendations</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data & pipeline
    _, train_df, _, customers_df, articles_df = load_datasets()
    pipeline = build_inference_pipeline(train_df)

    # Stats bar
    model_status = "Loaded" if pipeline.model_is_loaded else "Not Found"
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-value">{pipeline.num_users:,}</div>
            <div class="stat-label">Customers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{pipeline.num_items:,}</div>
            <div class="stat-label">Products</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(train_df):,}</div>
            <div class="stat-label">Transactions (Train)</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{model_status}</div>
            <div class="stat-label">Model Checkpoint</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -- Sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown("## Settings")

        sample_user_ids = customers_df["user_id"].head(200).to_list()

        input_method = st.radio(
            "Select input method",
            ["Choose from list", "Enter manually"],
            horizontal=True,
        )

        if input_method == "Choose from list":
            selected_user_id = st.selectbox(
                "Select Customer (user_id)",
                options=sample_user_ids,
                index=0,
                help="Showing first 200 customers from the dataset.",
            )
        else:
            selected_user_id = st.number_input(
                "Enter user_id",
                min_value=0,
                max_value=pipeline.num_users - 1,
                value=0,
                step=1,
            )

        recommendation_method = st.radio(
            "Recommendation Method",
            ["Hybrid Model (NCF + Visual)", "Popularity Baseline"],
            index=0 if pipeline.model_is_loaded else 1,
        )

        run_button = st.button(
            "Get Recommendations", use_container_width=True, type="primary"
        )

        st.markdown("---")
        st.markdown("### About")
        st.markdown(
            "This demo uses a **Hybrid NCF + Visual Feature** model "
            "trained on the H&M Personalized Fashion Recommendations dataset. "
            "The model combines collaborative filtering latent features with "
            "2048-dim image embeddings through dense fusion layers."
        )

    # -- Main content ------------------------------------------------------
    if run_button:
        # Customer profile
        customer_info = lookup_customer_profile(customers_df, int(selected_user_id))
        if customer_info:
            render_customer_profile(customer_info, int(selected_user_id))
        else:
            st.warning(
                f"Customer user_id={selected_user_id} not found in the dataset."
            )

        # Get recommendations
        use_hybrid = "Hybrid" in recommendation_method

        if use_hybrid and pipeline.model_is_loaded:
            with st.spinner("Running Hybrid Model inference ..."):
                recommendations = pipeline.recommend_hybrid(
                    user_id=int(selected_user_id)
                )
            st.success(
                f"Hybrid Model returned {len(recommendations)} recommendations."
            )
        else:
            if use_hybrid and not pipeline.model_is_loaded:
                st.info("No checkpoint found — falling back to Popularity Baseline.")
            recommendations = pipeline.recommend_popular()
            st.success(
                f"Popularity Baseline returned {len(recommendations)} recommendations."
            )

        # Display results
        st.markdown(
            f"### Top-{TOP_K_RECOMMENDATIONS} Recommended Products"
        )
        render_product_cards(recommendations, articles_df)
        render_details_table(recommendations, articles_df)

    else:
        # Landing prompt
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #8892b0;">
            <h3 style="color: #ccd6f6; font-weight: 600;">Select a customer and click "Get Recommendations"</h3>
            <p>Choose a customer from the sidebar, pick a recommendation method, and see their personalised top-12 products.</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

"""
app.py — Main Streamlit application for the Movie Recommender.

A hybrid movie recommendation system that provides:
  - SVD collaborative filtering for existing users (via scikit-surprise or scipy)
  - Genre/content-based recommendations for new users (cold start)

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd

# Local modules
from data_loader import (
    load_movies, load_ratings, load_links,
    get_movie_stats, get_unique_genres, get_sample_user_ids
)
from model import load_or_train_model, get_svd_recommendations, HAS_SURPRISE
from content_filter import get_cold_start_recommendations
from tmdb_client import get_poster_urls_batch, has_api_key


# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global Styles ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .app-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ffd200 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.25rem;
    }
    .app-header p {
        color: #b0b0cc;
        font-size: 1rem;
        font-weight: 300;
        margin-top: 0;
    }

    /* ── Movie Card ── */
    .movie-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 0;
        overflow: hidden;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .movie-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 40px rgba(233, 69, 96, 0.2);
        border-color: rgba(233, 69, 96, 0.3);
    }

    .movie-poster {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
        display: block;
    }

    .movie-info {
        padding: 0.8rem 0.9rem 1rem;
    }

    .movie-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e8e8f0;
        line-height: 1.3;
        margin-bottom: 0.5rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        min-height: 2.3em;
    }

    .movie-genres {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        margin-bottom: 0.55rem;
    }
    .genre-tag {
        background: rgba(240, 147, 251, 0.12);
        color: #d4a0f5;
        font-size: 0.65rem;
        font-weight: 500;
        padding: 2px 7px;
        border-radius: 20px;
        border: 1px solid rgba(240, 147, 251, 0.15);
        letter-spacing: 0.02em;
    }

    .movie-rating {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 6px;
    }
    .rating-stars {
        font-size: 0.88rem;
        font-weight: 600;
        color: #ffd200;
    }
    .rating-count {
        font-size: 0.7rem;
        color: #777799;
        font-weight: 400;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f093fb;
        font-weight: 600;
    }

    /* ── Info Boxes ── */
    .model-badge {
        display: inline-block;
        background: rgba(78, 204, 163, 0.12);
        color: #4ecca3;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(78, 204, 163, 0.2);
    }

    .api-warning {
        background: rgba(255, 210, 0, 0.08);
        border: 1px solid rgba(255, 210, 0, 0.2);
        border-radius: 10px;
        padding: 0.6rem 1rem;
        font-size: 0.8rem;
        color: #ccc;
        margin-bottom: 0.8rem;
    }
    .api-warning strong { color: #ffd200; }

    /* ── No Results ── */
    .empty-state {
        text-align: center;
        padding: 4rem 2rem;
        color: #777799;
    }
    .empty-state .icon {
        font-size: 3.5rem;
        margin-bottom: 1rem;
    }
    .empty-state h3 {
        color: #b0b0cc;
        font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ───────────────────────────────────────────────────────────────

movies_df = load_movies()
ratings_df = load_ratings()
links_df = load_links()
movies_with_stats = get_movie_stats(movies_df, ratings_df)
all_genres = get_unique_genres(movies_df)
sample_users = get_sample_user_ids(ratings_df)


# ─── Header ──────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <h1>🎬 Movie Recommender</h1>
    <p>Discover your next favorite film with AI-powered recommendations</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎯 Recommendation Mode")

    mode = st.radio(
        "Choose your mode:",
        ["🆕 New User (pick genres/favorites)", "👤 Existing User (browse by ID)"],
        index=0,
        help="New users get recommendations based on genre preferences. "
             "Existing users get personalized SVD predictions."
    )

    st.markdown("---")

    if "🆕" in mode:
        # ── New User Mode ──
        st.markdown("### 🎭 Your Preferences")

        selected_genres = st.multiselect(
            "Select favorite genres:",
            options=all_genres,
            default=None,
            placeholder="Choose genres...",
            help="Pick one or more genres you enjoy"
        )

        # Searchable movie selector
        movie_titles = sorted(movies_with_stats["title"].tolist())
        favorite_movies = st.multiselect(
            "Pick favorite movies (optional):",
            options=movie_titles,
            default=None,
            placeholder="Type to search...",
            max_selections=5,
            help="Select 1-5 movies you love. We'll find similar ones!"
        )

        selected_user_id = None

    else:
        # ── Existing User Mode ──
        st.markdown("### 👤 Select User")

        selected_user_id = st.selectbox(
            "User ID:",
            options=sample_users,
            help="These are real users from the MovieLens dataset. "
                 "Pick one to see personalized SVD recommendations."
        )

        # Show user stats
        if selected_user_id:
            user_ratings = ratings_df[ratings_df["userId"] == selected_user_id]
            st.caption(
                f"This user has rated **{len(user_ratings)}** movies "
                f"(avg: **{user_ratings['rating'].mean():.1f}** ⭐)"
            )

        selected_genres = []
        favorite_movies = []

    st.markdown("---")

    st.markdown("### ⚙️ Settings")
    num_recs = st.slider(
        "Number of recommendations:",
        min_value=5, max_value=20, value=10, step=1
    )

    # Model info
    st.markdown("---")
    st.markdown("### 📊 Model Info")

    model_type_label = "surprise SVD" if HAS_SURPRISE else "scipy SVD (fallback)"
    st.markdown(f'<span class="model-badge">🧠 {model_type_label}</span>', unsafe_allow_html=True)
    st.caption(f"Dataset: {len(movies_df):,} movies · {len(ratings_df):,} ratings")

    if not has_api_key():
        st.markdown(
            '<div class="api-warning">'
            '⚠️ <strong>TMDB API key not set</strong> — posters will show placeholders. '
            'See <code>.env.example</code> for setup.'
            '</div>',
            unsafe_allow_html=True
        )

    # Retrain button
    st.markdown("---")
    if st.button("🔄 Retrain Model", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    get_recs_button = st.button(
        "🚀 Get Recommendations",
        type="primary",
        use_container_width=True,
    )


# ─── Main Area — Recommendations ────────────────────────────────────────────

def render_movie_card(movie_row: pd.Series, poster_url: str):
    """Render a single movie card as HTML."""
    title = movie_row.get("title", "Unknown")
    genres = movie_row.get("genre_list", [])
    avg_rating = movie_row.get("avg_rating", 0.0)
    num_ratings = movie_row.get("num_ratings", 0)

    # Genre tags HTML
    genre_html = "".join(
        f'<span class="genre-tag">{g}</span>' for g in genres[:4]
    )
    if len(genres) > 4:
        genre_html += f'<span class="genre-tag">+{len(genres)-4}</span>'

    # Rating display
    rating_display = f"⭐ {avg_rating:.1f}" if avg_rating > 0 else "⭐ N/A"

    card_html = f"""
    <div class="movie-card">
        <img class="movie-poster" src="{poster_url}" alt="{title}" loading="lazy"
             onerror="this.onerror=null;this.src='data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%27500%27 height=%27750%27 viewBox=%270 0 500 750%27%3E%3Crect fill=%27%231a1a2e%27 width=%27500%27 height=%27750%27/%3E%3Ctext x=%27250%27 y=%27360%27 text-anchor=%27middle%27 fill=%27%23555577%27 font-family=%27sans-serif%27 font-size=%2720%27%3ENo Poster%3C/text%3E%3C/svg%3E'">
        <div class="movie-info">
            <div class="movie-title" title="{title}">{title}</div>
            <div class="movie-genres">{genre_html}</div>
            <div class="movie-rating">
                <span class="rating-stars">{rating_display}</span>
                <span class="rating-count">({num_ratings:,} ratings)</span>
            </div>
        </div>
    </div>
    """
    return card_html


def display_recommendations(recs_df: pd.DataFrame):
    """Display recommended movies in a responsive grid."""
    if recs_df.empty:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">🎬</div>
            <h3>No recommendations found</h3>
            <p>Try adjusting your preferences or selecting different genres.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Fetch poster URLs for all recommended movies
    movie_ids = recs_df["movieId"].tolist()
    poster_urls = get_poster_urls_batch(movie_ids, links_df)

    # Display in grid (5 columns)
    cols_per_row = 5
    rows = (len(recs_df) + cols_per_row - 1) // cols_per_row

    for row_idx in range(rows):
        cols = st.columns(cols_per_row, gap="medium")
        for col_idx in range(cols_per_row):
            item_idx = row_idx * cols_per_row + col_idx
            if item_idx < len(recs_df):
                movie = recs_df.iloc[item_idx]
                poster_url = poster_urls.get(movie["movieId"], "")
                with cols[col_idx]:
                    st.markdown(
                        render_movie_card(movie, poster_url),
                        unsafe_allow_html=True
                    )


# ─── Run Recommendations on Button Click ────────────────────────────────────

if get_recs_button:
    if "🆕" in mode:
        # ── New User: Cold Start ──
        if not selected_genres and not favorite_movies:
            st.warning(
                "🎭 Please select at least one genre or favorite movie to get recommendations!",
                icon="⚠️"
            )
        else:
            with st.spinner("✨ Finding movies you'll love..."):
                recs = get_cold_start_recommendations(
                    selected_genres, favorite_movies, movies_with_stats, n=num_recs
                )

                # Build description of what was used
                parts = []
                if selected_genres:
                    parts.append(f"genres: {', '.join(selected_genres)}")
                if favorite_movies:
                    parts.append(f"favorites: {', '.join(favorite_movies[:3])}")
                desc = " · ".join(parts)

                st.markdown(f"### 🎯 Recommendations based on {desc}")
                display_recommendations(recs)
    else:
        # ── Existing User: SVD ──
        if selected_user_id is None:
            st.warning("👤 Please select a user ID.", icon="⚠️")
        else:
            with st.spinner("🧠 Computing personalized recommendations..."):
                model, model_type = load_or_train_model(ratings_df)
                recs = get_svd_recommendations(
                    selected_user_id, model, model_type,
                    movies_with_stats, ratings_df, n=num_recs
                )

                st.markdown(
                    f"### 🎯 Personalized picks for User #{selected_user_id}"
                )
                if not recs.empty and "predicted_rating" in recs.columns:
                    avg_pred = recs["predicted_rating"].mean()
                    st.caption(
                        f"Average predicted rating: ⭐ {avg_pred:.2f} / 5.0 "
                        f"· Model: {model_type}"
                    )
                display_recommendations(recs)
else:
    # ── Welcome State ──
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🍿</div>
        <h3>Ready to discover great movies?</h3>
        <p>Configure your preferences in the sidebar, then click <strong>Get Recommendations</strong>!</p>
    </div>
    """, unsafe_allow_html=True)

    # Show a quick overview of the dataset
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎬 Movies", f"{len(movies_df):,}")
    with col2:
        st.metric("⭐ Ratings", f"{len(ratings_df):,}")
    with col3:
        st.metric("👥 Users", f"{ratings_df['userId'].nunique():,}")
    with col4:
        st.metric("🎭 Genres", f"{len(all_genres)}")

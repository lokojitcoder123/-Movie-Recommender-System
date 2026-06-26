"""
data_loader.py — Download, unzip, and load the MovieLens ml-latest-small dataset.

Handles automatic downloading on first run, caching with @st.cache_data,
and computing per-movie aggregate statistics (average rating, Bayesian average).
"""

import os
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import streamlit as st

# ─── Constants ───────────────────────────────────────────────────────────────

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ZIP_PATH = os.path.join(DATA_DIR, "ml-latest-small.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "ml-latest-small")


# ─── Download & Extract ─────────────────────────────────────────────────────

def _download_dataset():
    """Download the MovieLens ml-latest-small zip if it hasn't been downloaded yet."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check if the extracted CSV files already exist
    if os.path.exists(os.path.join(EXTRACTED_DIR, "movies.csv")):
        return  # Already downloaded and extracted

    # Download the zip file
    if not os.path.exists(ZIP_PATH):
        st.info("📥 Downloading MovieLens dataset (first run only)...")
        try:
            urllib.request.urlretrieve(MOVIELENS_URL, ZIP_PATH)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download MovieLens dataset from {MOVIELENS_URL}. "
                f"Check your internet connection.\nError: {e}"
            )

    # Extract the zip file
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(DATA_DIR)
        # Clean up the zip to save disk space
        os.remove(ZIP_PATH)
    except zipfile.BadZipFile:
        # Corrupted download — remove and let user retry
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        raise RuntimeError(
            "Downloaded zip file was corrupted. Please restart the app to retry."
        )


# ─── Data Loading (Cached) ──────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading movie data...")
def load_movies() -> pd.DataFrame:
    """Load and return movies.csv with cleaned genre lists."""
    _download_dataset()
    movies = pd.read_csv(os.path.join(EXTRACTED_DIR, "movies.csv"))
    # Create a list-of-genres column for easier filtering
    movies["genre_list"] = movies["genres"].apply(
        lambda g: g.split("|") if g != "(no genres listed)" else []
    )
    return movies


@st.cache_data(show_spinner="Loading ratings data...")
def load_ratings() -> pd.DataFrame:
    """Load and return ratings.csv."""
    _download_dataset()
    return pd.read_csv(os.path.join(EXTRACTED_DIR, "ratings.csv"))


@st.cache_data(show_spinner="Loading movie links...")
def load_links() -> pd.DataFrame:
    """Load and return links.csv (maps movieId → imdbId, tmdbId)."""
    _download_dataset()
    links = pd.read_csv(os.path.join(EXTRACTED_DIR, "links.csv"))
    # tmdbId can have NaN values; keep as float for now, convert when needed
    return links


# ─── Aggregate Statistics ────────────────────────────────────────────────────

@st.cache_data(show_spinner="Computing movie statistics...")
def get_movie_stats(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-movie aggregate stats and merge into the movies DataFrame.

    Adds columns:
      - avg_rating:  mean rating for each movie
      - num_ratings: number of ratings for each movie
      - bayesian_avg: Bayesian-smoothed average (avoids noisy ratings from
                      movies with very few reviews)

    Bayesian formula:
        bayesian_avg = (avg_rating * num_ratings + C * global_avg) / (num_ratings + C)
    where C is the median number of ratings across all movies.
    """
    # Compute per-movie stats
    stats = ratings.groupby("movieId")["rating"].agg(
        avg_rating="mean",
        num_ratings="count"
    ).reset_index()

    # Bayesian smoothing parameters
    global_avg = ratings["rating"].mean()
    C = stats["num_ratings"].median()  # Prior strength = median vote count

    stats["bayesian_avg"] = (
        (stats["avg_rating"] * stats["num_ratings"] + C * global_avg)
        / (stats["num_ratings"] + C)
    )

    # Merge stats into movies
    movies_with_stats = movies.merge(stats, on="movieId", how="left")
    # Fill movies with no ratings
    movies_with_stats["avg_rating"] = movies_with_stats["avg_rating"].fillna(0.0)
    movies_with_stats["num_ratings"] = movies_with_stats["num_ratings"].fillna(0).astype(int)
    movies_with_stats["bayesian_avg"] = movies_with_stats["bayesian_avg"].fillna(0.0)

    return movies_with_stats


# ─── Helper Functions ────────────────────────────────────────────────────────

@st.cache_data
def get_unique_genres(movies: pd.DataFrame) -> list:
    """Extract a sorted list of all unique genres from the dataset."""
    all_genres = set()
    for genre_list in movies["genre_list"]:
        all_genres.update(genre_list)
    # Remove empty strings if any
    all_genres.discard("")
    return sorted(all_genres)


@st.cache_data
def get_sample_user_ids(ratings: pd.DataFrame, n: int = 50) -> list:
    """
    Return a sample of user IDs that have a reasonable number of ratings.
    Picks users with at least 20 ratings for meaningful SVD predictions.
    """
    user_counts = ratings.groupby("userId").size()
    # Filter users with at least 20 ratings
    active_users = user_counts[user_counts >= 20].index.tolist()
    # Sample up to n users
    rng = np.random.RandomState(42)  # Reproducible sampling
    if len(active_users) > n:
        sampled = rng.choice(active_users, size=n, replace=False).tolist()
    else:
        sampled = active_users
    return sorted(sampled)

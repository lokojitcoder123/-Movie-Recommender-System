"""
content_filter.py — Genre/content-based cold-start recommendation logic.

For new users with no rating history, this module provides recommendations
based on genre preferences and/or favorite movie titles using:
  - TF-IDF vectorization on movie genres
  - Cosine similarity for finding similar movies
  - Bayesian-averaged ratings for quality ranking
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


# ─── TF-IDF Model (Cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Building genre similarity model...")
def _build_tfidf_model(genre_strings: tuple):
    """
    Build a TF-IDF matrix from genre strings.

    Args:
        genre_strings: Tuple of genre strings (pipe-separated converted to space-separated).
                       Must be a tuple (not list) for st.cache_resource hashing.

    Returns:
        (tfidf_matrix, vectorizer) — the TF-IDF matrix and fitted vectorizer.
    """
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w[\w'-]+\b")
    tfidf_matrix = vectorizer.fit_transform(genre_strings)
    return tfidf_matrix, vectorizer


def _get_tfidf_matrix(movies_df: pd.DataFrame):
    """Get or build the TF-IDF matrix for movie genres."""
    # Convert pipe-separated genres to space-separated for TF-IDF
    genre_strings = tuple(
        movies_df["genres"].str.replace("|", " ", regex=False)
        .str.replace("(no genres listed)", "", regex=False)
        .tolist()
    )
    return _build_tfidf_model(genre_strings)


# ─── Genre-Based Recommendations ────────────────────────────────────────────

def get_genre_recommendations(
    selected_genres: list,
    movies_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Recommend movies based on selected genre preferences.

    Scores each movie by how well its genres match the selected genres,
    weighted by the movie's Bayesian-averaged rating to prefer
    well-rated, well-known movies.

    Args:
        selected_genres: List of genre strings (e.g., ["Action", "Comedy"]).
        movies_df:       DataFrame with movie info + stats columns
                         (avg_rating, num_ratings, bayesian_avg).
        n:               Number of recommendations to return.

    Returns:
        DataFrame of top-N recommended movies.
    """
    if not selected_genres:
        return pd.DataFrame()

    # Score each movie by genre overlap
    selected_set = set(selected_genres)

    def genre_match_score(genre_list):
        if not genre_list:
            return 0.0
        matching = len(set(genre_list) & selected_set)
        total = len(genre_list)
        # Weighted Jaccard-like: proportion of selected genres covered
        # + proportion of movie's genres that match (to prefer focused movies)
        if total == 0:
            return 0.0
        precision = matching / total  # How focused the movie is on selected genres
        recall = matching / len(selected_set)  # How many selected genres are covered
        if precision + recall == 0:
            return 0.0
        # F1-like score
        return 2 * precision * recall / (precision + recall)

    scores = movies_df["genre_list"].apply(genre_match_score)

    # Combine genre match with Bayesian average rating
    # Normalize Bayesian avg to [0, 1] range for combining
    max_bayesian = movies_df["bayesian_avg"].max()
    if max_bayesian > 0:
        quality_score = movies_df["bayesian_avg"] / max_bayesian
    else:
        quality_score = 0.0

    # Final score: 60% genre match + 40% quality
    combined_score = 0.6 * scores + 0.4 * quality_score

    # Filter out movies with zero genre match
    mask = scores > 0
    result = movies_df[mask].copy()
    result["recommendation_score"] = combined_score[mask]

    # Sort by combined score descending
    result = result.sort_values("recommendation_score", ascending=False).head(n)

    return result.reset_index(drop=True)


# ─── Similar Movies (by Content) ────────────────────────────────────────────

def get_similar_movies(
    favorite_titles: list,
    movies_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Find movies most similar to the user's favorite movies based on
    genre cosine similarity (TF-IDF).

    Args:
        favorite_titles: List of movie title strings that the user picked.
        movies_df:       DataFrame with movie info + stats columns.
        n:               Number of recommendations to return.

    Returns:
        DataFrame of top-N similar movies (excluding the favorites themselves).
    """
    if not favorite_titles:
        return pd.DataFrame()

    tfidf_matrix, _ = _get_tfidf_matrix(movies_df)

    # Find indices of the favorite movies
    fav_indices = []
    for title in favorite_titles:
        matches = movies_df[movies_df["title"] == title].index.tolist()
        if matches:
            fav_indices.append(matches[0])

    if not fav_indices:
        return pd.DataFrame()

    # Compute cosine similarity between favorites and all movies
    # Average the similarity vectors across all favorites
    fav_vectors = tfidf_matrix[fav_indices]
    sim_scores = cosine_similarity(fav_vectors, tfidf_matrix)

    # Average similarity across all favorite movies
    avg_sim = sim_scores.mean(axis=0)

    # Boost by Bayesian average (quality signal)
    max_bayesian = movies_df["bayesian_avg"].max()
    if max_bayesian > 0:
        quality_bonus = movies_df["bayesian_avg"].values / max_bayesian
    else:
        quality_bonus = np.zeros(len(movies_df))

    # Combined score: 70% similarity + 30% quality
    combined = 0.7 * avg_sim + 0.3 * quality_bonus

    # Exclude the favorites themselves
    fav_movie_ids = set(movies_df.iloc[fav_indices]["movieId"].tolist())
    mask = ~movies_df["movieId"].isin(fav_movie_ids)

    result = movies_df[mask].copy()
    result["similarity_score"] = combined[mask.values]

    # Sort by combined score descending
    result = result.sort_values("similarity_score", ascending=False).head(n)

    return result.reset_index(drop=True)


# ─── Combined Cold Start ────────────────────────────────────────────────────

def get_cold_start_recommendations(
    selected_genres: list,
    favorite_titles: list,
    movies_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Combine genre-based and similarity-based recommendations for cold-start users.

    If the user provided both genres and favorite movies, merge results.
    If only one is provided, use that approach alone.

    Args:
        selected_genres:  List of genre strings (may be empty).
        favorite_titles:  List of movie title strings (may be empty).
        movies_df:        DataFrame with movie info + stats columns.
        n:                Number of recommendations to return.

    Returns:
        DataFrame of top-N recommended movies.
    """
    has_genres = bool(selected_genres)
    has_favorites = bool(favorite_titles)

    if not has_genres and not has_favorites:
        return pd.DataFrame()

    if has_genres and has_favorites:
        # Get more from each, then merge and deduplicate
        genre_recs = get_genre_recommendations(selected_genres, movies_df, n=n * 2)
        similar_recs = get_similar_movies(favorite_titles, movies_df, n=n * 2)

        if genre_recs.empty:
            return similar_recs.head(n).reset_index(drop=True)
        if similar_recs.empty:
            return genre_recs.head(n).reset_index(drop=True)

        # Merge: rank-based fusion
        # Assign rank scores (higher = better)
        genre_recs = genre_recs.copy()
        genre_recs["genre_rank"] = range(len(genre_recs), 0, -1)

        similar_recs = similar_recs.copy()
        similar_recs["sim_rank"] = range(len(similar_recs), 0, -1)

        # Outer merge on movieId
        merged = genre_recs[["movieId", "genre_rank"]].merge(
            similar_recs[["movieId", "sim_rank"]],
            on="movieId", how="outer"
        )
        merged["genre_rank"] = merged["genre_rank"].fillna(0)
        merged["sim_rank"] = merged["sim_rank"].fillna(0)
        merged["combined_rank"] = merged["genre_rank"] + merged["sim_rank"]

        # Sort by combined rank
        merged = merged.sort_values("combined_rank", ascending=False).head(n)

        # Look up full movie info
        result = movies_df[movies_df["movieId"].isin(merged["movieId"])].copy()
        # Preserve the merged ordering
        result = result.set_index("movieId").loc[merged["movieId"].values].reset_index()

        return result.head(n).reset_index(drop=True)

    elif has_genres:
        return get_genre_recommendations(selected_genres, movies_df, n=n)

    else:  # has_favorites only
        return get_similar_movies(favorite_titles, movies_df, n=n)

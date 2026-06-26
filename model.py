"""
model.py — SVD-based collaborative filtering for movie recommendations.

Primary: Uses the `surprise` library's SVD algorithm.
Fallback: If `surprise` is not installed, uses scipy.sparse.linalg.svds
          on a user-item rating matrix for a pure-numpy/scipy implementation.

The trained model is cached to disk (pickle) so it doesn't retrain on every run.
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "svd_model.pkl")
SCIPY_MODEL_PATH = os.path.join(DATA_DIR, "scipy_svd_model.pkl")

# Try importing surprise; set a flag for fallback
try:
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import cross_validate
    HAS_SURPRISE = True
except ImportError:
    HAS_SURPRISE = False


# ─── Surprise SVD ────────────────────────────────────────────────────────────

def _train_surprise_svd(ratings_df: pd.DataFrame) -> "SVD":
    """Train an SVD model using the surprise library and save to disk."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        ratings_df[["userId", "movieId", "rating"]], reader
    )

    # Train on the full dataset (no test split — we want best predictions)
    trainset = data.build_full_trainset()

    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    model.fit(trainset)

    # Save to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def _load_surprise_svd() -> "SVD":
    """Load a previously trained surprise SVD model from disk."""
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _predict_surprise(model, user_id: int, movie_ids: list) -> list:
    """
    Predict ratings for a list of movie IDs using the surprise SVD model.
    Returns a list of (movieId, predicted_rating) tuples.
    """
    predictions = []
    for mid in movie_ids:
        pred = model.predict(user_id, mid)
        predictions.append((mid, pred.est))
    return predictions


# ─── Scipy SVD Fallback ─────────────────────────────────────────────────────

def _train_scipy_svd(ratings_df: pd.DataFrame) -> dict:
    """
    Fallback SVD using scipy.sparse.linalg.svds.
    Builds a user-item matrix, decomposes it, and stores the components.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    # Create user/movie index mappings
    user_ids = ratings_df["userId"].unique()
    movie_ids = ratings_df["movieId"].unique()
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_to_idx = {mid: i for i, mid in enumerate(movie_ids)}
    idx_to_movie = {i: mid for mid, i in movie_to_idx.items()}

    # Build sparse user-item matrix
    n_users = len(user_ids)
    n_movies = len(movie_ids)
    rows = ratings_df["userId"].map(user_to_idx).values
    cols = ratings_df["movieId"].map(movie_to_idx).values
    vals = ratings_df["rating"].values

    R = csr_matrix((vals, (rows, cols)), shape=(n_users, n_movies))

    # Compute per-user mean for centering
    user_means = np.array(R.sum(axis=1)).flatten()
    user_counts = np.array((R != 0).sum(axis=1)).flatten()
    user_means = np.divide(
        user_means, user_counts,
        out=np.zeros_like(user_means),
        where=user_counts != 0
    )

    # Center the matrix (subtract user mean from non-zero entries)
    R_dense = R.toarray().astype(np.float64)
    for i in range(n_users):
        mask = R_dense[i, :] != 0
        R_dense[i, mask] -= user_means[i]

    # SVD decomposition — use k=50 factors (less than min dimension)
    k = min(50, min(n_users, n_movies) - 1)
    U, sigma, Vt = svds(csr_matrix(R_dense), k=k)

    # Reconstruct predicted ratings = U * diag(sigma) * Vt + user_means
    sigma_diag = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma_diag), Vt)
    # Add back user means
    predicted += user_means[:, np.newaxis]

    model_data = {
        "predicted": predicted,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx,
        "idx_to_movie": idx_to_movie,
        "user_ids": user_ids.tolist(),
        "movie_ids": movie_ids.tolist(),
    }

    # Save to disk
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SCIPY_MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    return model_data


def _load_scipy_svd() -> dict:
    """Load a previously trained scipy SVD model from disk."""
    with open(SCIPY_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _predict_scipy(model_data: dict, user_id: int, movie_ids: list) -> list:
    """
    Predict ratings for a list of movie IDs using the scipy SVD model.
    Returns a list of (movieId, predicted_rating) tuples.
    """
    user_to_idx = model_data["user_to_idx"]
    movie_to_idx = model_data["movie_to_idx"]
    predicted = model_data["predicted"]

    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]
    predictions = []
    for mid in movie_ids:
        if mid in movie_to_idx:
            movie_idx = movie_to_idx[mid]
            pred_rating = float(predicted[user_idx, movie_idx])
            # Clip to valid rating range
            pred_rating = max(0.5, min(5.0, pred_rating))
            predictions.append((mid, pred_rating))
    return predictions


# ─── Public API ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training recommendation model (one-time)...")
def load_or_train_model(ratings_df: pd.DataFrame, force_retrain: bool = False):
    """
    Load the trained SVD model from disk, or train a new one.

    Returns a tuple of (model_object, model_type_string).
    model_type is either 'surprise' or 'scipy'.
    """
    if HAS_SURPRISE:
        if not force_retrain and os.path.exists(MODEL_PATH):
            try:
                model = _load_surprise_svd()
                return model, "surprise"
            except Exception:
                pass  # Corrupted file — retrain
        model = _train_surprise_svd(ratings_df)
        return model, "surprise"
    else:
        # Fallback to scipy SVD
        if not force_retrain and os.path.exists(SCIPY_MODEL_PATH):
            try:
                model = _load_scipy_svd()
                return model, "scipy"
            except Exception:
                pass
        model = _train_scipy_svd(ratings_df)
        return model, "scipy"


def get_svd_recommendations(
    user_id: int,
    model,
    model_type: str,
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n: int = 10
) -> pd.DataFrame:
    """
    Get top-N movie recommendations for an existing user via SVD.

    Predicts ratings for all movies the user hasn't rated yet,
    then returns the top-N highest predicted-rating movies.

    Parameters:
        user_id:    The user ID from the dataset.
        model:      The trained SVD model (surprise or scipy dict).
        model_type: 'surprise' or 'scipy'.
        movies_df:  DataFrame with movie info (must include stats columns).
        ratings_df: Full ratings DataFrame.
        n:          Number of recommendations to return.

    Returns:
        DataFrame with recommended movies, including predicted_rating column.
    """
    # Find movies the user has already rated
    rated_movie_ids = set(
        ratings_df[ratings_df["userId"] == user_id]["movieId"].tolist()
    )
    # Get all unrated movies
    all_movie_ids = movies_df["movieId"].tolist()
    unrated_ids = [mid for mid in all_movie_ids if mid not in rated_movie_ids]

    if not unrated_ids:
        return pd.DataFrame()

    # Predict ratings for unrated movies
    if model_type == "surprise":
        preds = _predict_surprise(model, user_id, unrated_ids)
    else:
        preds = _predict_scipy(model, user_id, unrated_ids)

    if not preds:
        return pd.DataFrame()

    # Sort by predicted rating descending, take top-N
    preds.sort(key=lambda x: x[1], reverse=True)
    top_preds = preds[:n]

    # Build result DataFrame
    rec_ids = [p[0] for p in top_preds]
    pred_ratings = {p[0]: p[1] for p in top_preds}

    recs = movies_df[movies_df["movieId"].isin(rec_ids)].copy()
    recs["predicted_rating"] = recs["movieId"].map(pred_ratings)
    recs = recs.sort_values("predicted_rating", ascending=False).head(n)

    return recs.reset_index(drop=True)

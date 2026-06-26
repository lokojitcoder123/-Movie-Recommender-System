"""
tmdb_client.py — TMDB API integration for fetching movie poster images.

Reads the TMDB API key from environment variables (via .env file).
Caches poster URLs to disk so repeated runs don't re-hit the API.
Gracefully falls back to a placeholder image on any failure.
"""

import os
import json
import requests
import streamlit as st
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ─── Constants ───────────────────────────────────────────────────────────────

def _get_tmdb_api_key() -> str:
    """
    Resolve the TMDB API key from multiple sources (priority order):
      1. Environment variable (via .env or system)
      2. Streamlit secrets  (for Streamlit Cloud deployments)
    """
    # 1. Environment variable (.env / system)
    key = os.getenv("TMDB_API_KEY", "")
    if key and key != "your_tmdb_api_key_here":
        return key

    # 2. Streamlit secrets (Streamlit Cloud)
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
        if key and key != "your_tmdb_api_key_here":
            return key
    except Exception:
        pass  # st.secrets not available or not configured

    return ""


TMDB_API_KEY = _get_tmdb_api_key()
TMDB_BASE_URL = "https://api.themoviedb.org/3/movie"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Placeholder image when poster is unavailable (inline SVG data URI — always works)
PLACEHOLDER_POSTER = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='500' height='750' "
    "viewBox='0 0 500 750'%3E%3Crect fill='%231a1a2e' width='500' height='750'/%3E"
    "%3Ctext x='250' y='340' text-anchor='middle' fill='%23555577' font-family='sans-serif' "
    "font-size='28'%3E🎬%3C/text%3E%3Ctext x='250' y='390' text-anchor='middle' "
    "fill='%23555577' font-family='sans-serif' font-size='18'%3ENo Poster%3C/text%3E"
    "%3Ctext x='250' y='420' text-anchor='middle' fill='%23444466' font-family='sans-serif' "
    "font-size='14'%3ESet TMDB API key%3C/text%3E%3Ctext x='250' y='442' "
    "text-anchor='middle' fill='%23444466' font-family='sans-serif' "
    "font-size='14'%3Ein .env file%3C/text%3E%3C/svg%3E"
)

# Disk cache for poster URLs
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
POSTER_CACHE_FILE = os.path.join(DATA_DIR, "poster_cache.json")


# ─── Poster Cache (Disk-backed) ─────────────────────────────────────────────

def _load_poster_cache() -> dict:
    """Load the poster URL cache from disk."""
    if os.path.exists(POSTER_CACHE_FILE):
        try:
            with open(POSTER_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_poster_cache(cache: dict):
    """Save the poster URL cache to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)
    try:
        with open(POSTER_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except IOError:
        pass  # Non-critical — just skip caching


# ─── TMDB API ───────────────────────────────────────────────────────────────

def _fetch_poster_from_tmdb(tmdb_id: int) -> str:
    """
    Fetch the poster URL from TMDB API for a given TMDB movie ID.

    Returns the full poster URL or the placeholder if anything fails.
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_api_key_here":
        return PLACEHOLDER_POSTER

    try:
        url = f"{TMDB_BASE_URL}/{int(tmdb_id)}"
        response = requests.get(
            url,
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")
        if poster_path:
            return f"{TMDB_IMAGE_BASE}{poster_path}"
        else:
            return PLACEHOLDER_POSTER

    except (requests.RequestException, ValueError, KeyError):
        # Network error, invalid JSON, missing field — all fall back gracefully
        return PLACEHOLDER_POSTER


# ─── Public API ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=86400)  # Cache for 24 hours in Streamlit
def get_poster_url(tmdb_id) -> str:
    """
    Get the poster URL for a movie given its TMDB ID.

    Checks disk cache first, then calls the TMDB API if needed.
    Always returns a valid URL (poster or placeholder).

    Args:
        tmdb_id: The TMDB movie ID (from links.csv). May be NaN or None.

    Returns:
        Full URL string for the poster image.
    """
    # Handle missing/invalid TMDB IDs
    if tmdb_id is None or (isinstance(tmdb_id, float) and (tmdb_id != tmdb_id)):  # NaN check
        return PLACEHOLDER_POSTER

    tmdb_id_str = str(int(tmdb_id))

    # Check disk cache
    cache = _load_poster_cache()
    if tmdb_id_str in cache:
        return cache[tmdb_id_str]

    # Fetch from TMDB
    poster_url = _fetch_poster_from_tmdb(int(tmdb_id))

    # Save to cache (even if placeholder, to avoid re-fetching known failures)
    cache[tmdb_id_str] = poster_url
    _save_poster_cache(cache)

    return poster_url


def get_poster_urls_batch(movie_ids: list, links_df) -> dict:
    """
    Get poster URLs for a batch of movie IDs.

    Args:
        movie_ids: List of MovieLens movie IDs.
        links_df:  DataFrame with movieId → tmdbId mapping.

    Returns:
        Dict mapping movieId → poster_url.
    """
    # Build movieId → tmdbId lookup
    tmdb_lookup = dict(
        zip(links_df["movieId"], links_df["tmdbId"])
    )

    result = {}
    for mid in movie_ids:
        tmdb_id = tmdb_lookup.get(mid)
        result[mid] = get_poster_url(tmdb_id)

    return result


def has_api_key() -> bool:
    """Check if a valid TMDB API key is configured."""
    return bool(TMDB_API_KEY) and TMDB_API_KEY != "your_tmdb_api_key_here"

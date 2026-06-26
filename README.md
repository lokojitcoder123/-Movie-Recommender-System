# 🎬 Movie Recommender

A hybrid movie recommendation web app built with **Streamlit** and **Python**. It uses **SVD collaborative filtering** (via scikit-surprise) for existing users and **TF-IDF genre-based content filtering** for new users, trained on the [MovieLens ml-latest-small](https://grouplens.org/datasets/movielens/latest/) dataset. Movie posters are fetched from the TMDB API.

---

## 🚀 Quick Start

### 1. Clone & Create Virtual Environment

```bash
cd "Movie Recomendetion System"
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # macOS/Linux
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **⚠️ Note for Windows users:** `scikit-surprise` requires C++ Build Tools.
> If installation fails, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first.
> Alternatively, the app includes a **scipy-based SVD fallback** that works without surprise — just remove `scikit-surprise` from `requirements.txt`.

### 3. Set Up TMDB API Key (Optional but Recommended)

Movie posters require a free TMDB API key:

1. Create a free account at [TMDB](https://www.themoviedb.org/signup)
2. Go to **Settings → API → Create → Developer**
3. Copy your **API Key (v3 auth)**
4. Create a `.env` file in the project root:

```bash
cp .env.example .env
# Then edit .env and paste your key
```

> Without a TMDB key, the app still works — posters will show placeholder images.

### 4. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` 🎉

---

## 🧠 How It Works

### Hybrid Recommendation Approach

| Mode | Algorithm | When Used |
|------|-----------|-----------|
| **New User** | TF-IDF + Cosine Similarity on genres, weighted by Bayesian-averaged ratings | User selects genres and/or favorite movies |
| **Existing User** | SVD Collaborative Filtering (surprise/scipy) | User picks a real userId from the dataset |

**New User (Cold Start):**
- Select favorite genres → movies are ranked by genre overlap × quality score
- Pick favorite movies → TF-IDF cosine similarity finds similar movies
- Both → rank fusion combines genre and similarity signals

**Existing User (SVD):**
- The SVD model predicts ratings for all movies the user hasn't rated
- Top-N highest predicted ratings are returned as recommendations

### Bayesian Average Rating

To avoid noisy recommendations from movies with very few ratings, the app uses a Bayesian-smoothed average:

```
bayesian_avg = (avg_rating × num_ratings + C × global_avg) / (num_ratings + C)
```

where `C` is the median number of ratings across all movies.

---

## 📁 Project Structure

```
Movie Recomendetion System/
├── app.py                  # Main Streamlit app (UI + routing)
├── data_loader.py          # Download/load MovieLens CSVs, compute stats
├── model.py                # SVD training/loading (surprise + scipy fallback)
├── content_filter.py       # Genre/content-based cold-start logic (TF-IDF)
├── tmdb_client.py          # TMDB API calls + poster caching + fallback
├── requirements.txt        # Python dependencies
├── .env.example            # Template for TMDB API key
├── .gitignore              # Excludes .env, data/, *.pkl, __pycache__
├── README.md               # This file
└── data/                   # Auto-downloaded MovieLens CSVs (gitignored)
```

---

## ⚠️ Known Limitations

- **Cold start**: New users only get genre-based recommendations (no collaborative signal)
- **Small dataset**: MovieLens ml-latest-small has ~100K ratings across ~9,700 movies — production systems use millions
- **TMDB rate limits**: The TMDB API allows ~40 requests per 10 seconds; poster URLs are cached to disk to minimize API calls
- **Surprise installation**: May require C++ build tools on Windows; the scipy fallback is less optimized but works everywhere
- **No real-time learning**: The SVD model is trained once and cached — it doesn't update with new ratings

---

## 📄 License

This project uses the [MovieLens dataset](https://grouplens.org/datasets/movielens/) under its own terms.
TMDB data is provided by [The Movie Database (TMDB)](https://www.themoviedb.org/).

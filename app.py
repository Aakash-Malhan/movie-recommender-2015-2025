import os, re, zipfile, urllib.request, pickle
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import gradio as gr

# ---------------------------
# Config / caching locations
# ---------------------------
CACHE_DIR = Path.home() / ".cache" / "recsys25m"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "mf_model.pkl"
MOVIES_PATH = ARTIFACTS_DIR / "movies_2015_2025.parquet"
RATINGS_META_PATH = ARTIFACTS_DIR / "ratings_meta.txt"  # quick info for README/logs

ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
YEAR_RE = re.compile(r"\((\d{4})\)$")

# Subset knobs (tune for startup speed vs quality)
YEAR_MIN, YEAR_MAX = 2015, 2025
MIN_USER_RATINGS = 10
MAX_USERS = 30_000           # keep top-N most active users to speed up CPU training
FAST_MODE_MAX_RATINGS = 800_000  # cap total rating rows for first build
N_COMPONENTS = 100           # latent factors

# ---------------------------
# Helpers: dataset loading
# ---------------------------
def _download_ml25m() -> Path:
    zpath = CACHE_DIR / "ml-25m.zip"
    if not zpath.exists():
        print("Downloading MovieLens 25M (~250MB)…")
        urllib.request.urlretrieve(ML25M_URL, zpath)
    return zpath

def _find_csv_root() -> Path:
    # Search for folder containing movies.csv & ratings.csv (handles nested ml-25m/ml-25m)
    for p in CACHE_DIR.rglob("movies.csv"):
        cand = p.parent
        if (cand / "ratings.csv").exists():
            return cand
    # Not extracted yet → extract now and search again
    with zipfile.ZipFile(_download_ml25m(), "r") as z:
        z.extractall(CACHE_DIR)
    for p in CACHE_DIR.rglob("movies.csv"):
        cand = p.parent
        if (cand / "ratings.csv").exists():
            return cand
    raise FileNotFoundError("Could not locate movies.csv/ratings.csv after extraction.")

def _year_from_title(title: str) -> Optional[int]:
    m = YEAR_RE.search(title or "")
    return int(m.group(1)) if m else None

def load_ml25m_2015_2025() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ratings, movies) filtered to 2015–2025 with speed caps."""
    root = _find_csv_root()

    movies = pd.read_csv(root / "movies.csv", usecols=["movieId", "title", "genres"])
    movies["year"] = movies["title"].map(_year_from_title)
    movies = movies.dropna(subset=["year"]).astype({"year": "int16"})
    movies = movies[(movies["year"] >= YEAR_MIN) & (movies["year"] <= YEAR_MAX)]
    movies = movies.rename(columns={"movieId": "item_id"})

    ratings = pd.read_csv(
        root / "ratings.csv",
        usecols=["userId", "movieId", "rating", "timestamp"],
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"userId": "user_id", "movieId": "item_id"})

    ratings = ratings[ratings["item_id"].isin(movies["item_id"])]

    # Active users only (after year filter)
    user_counts = ratings["user_id"].value_counts()
    active_users = user_counts[user_counts >= MIN_USER_RATINGS].index
    ratings = ratings[ratings["user_id"].isin(active_users)]

    # Cap to most-active users for speed
    if MAX_USERS is not None and len(active_users) > MAX_USERS:
        top_users = user_counts.loc[active_users].nlargest(MAX_USERS).index
        ratings = ratings[ratings["user_id"].isin(top_users)]

    # Optional cap on total rows (random sample)
    if FAST_MODE_MAX_RATINGS is not None and len(ratings) > FAST_MODE_MAX_RATINGS:
        ratings = ratings.sample(FAST_MODE_MAX_RATINGS, random_state=42)

    # Keep only movies that still have ratings
    movies = movies[movies["item_id"].isin(ratings["item_id"].unique())]

    # Persist quick artifacts (speeds up reloads)
    movies.to_parquet(MOVIES_PATH, index=False)
    with open(RATINGS_META_PATH, "w") as f:
        f.write(f"rows={len(ratings)}, users={ratings.user_id.nunique()}, movies={len(movies)}\n")

    return ratings[["user_id", "item_id", "rating", "timestamp"]], movies[["item_id", "title", "year", "genres"]]

# ---------------------------
# Recommender (TruncatedSVD)
# ---------------------------
class MFRecommender:
    def __init__(self, n_components=100, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        self.global_mean = 0.0
        self.user_index = {}
        self.item_index = {}
        self.index_user = {}
        self.index_item = {}
        self.U = None
        self.V = None

    def _build_maps(self, ratings: pd.DataFrame):
        users = ratings["user_id"].astype(int).unique()
        items = ratings["item_id"].astype(int).unique()
        self.user_index = {u: i for i, u in enumerate(sorted(users))}
        self.item_index = {m: i for i, m in enumerate(sorted(items))}
        self.index_user = {i: u for u, i in self.user_index.items()}
        self.index_item = {i: m for m, i in self.item_index.items()}

    def _to_sparse(self, ratings: pd.DataFrame) -> csr_matrix:
        rows = ratings["user_id"].map(self.user_index).to_numpy()
        cols = ratings["item_id"].map(self.item_index).to_numpy()
        data = ratings["rating"].astype(float).to_numpy()
        return csr_matrix((data, (rows, cols)), shape=(len(self.user_index), len(self.item_index)))

    def fit(self, ratings: pd.DataFrame):
        self._build_maps(ratings)
        R = self._to_sparse(ratings).astype(np.float32)
        self.global_mean = (R.data.mean() if R.nnz > 0 else 0.0)
        R = R.copy()
        if R.nnz > 0:
            R.data = R.data - self.global_mean
        self.svd.fit(R)
        V = self.svd.components_.T
        U = R @ V
        # Normalize for stability
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
        self.V, self.U = V, U
        return self

    def recommend_from_liked(self, liked_item_ids: List[int], topk: int = 10, exclude_item_ids=None):
        liked_idx = [self.item_index[i] for i in liked_item_ids if i in self.item_index]
        if not liked_idx:
            scores = np.zeros(len(self.index_item)) + self.global_mean
            order = np.argsort(-scores)
        else:
            u_vec = self.V[liked_idx, :].mean(axis=0)
            u_vec = u_vec / (np.linalg.norm(u_vec) + 1e-8)
            scores = u_vec @ self.V.T + self.global_mean
            order = np.argsort(-scores)
        exclude = set([self.item_index[i] for i in (exclude_item_ids or []) if i in self.item_index])
        order = [i for i in order if i not in exclude]
        top = order[:topk]
        return [ (self.index_item[i], float(scores[i])) for i in top ]

# ---------------------------
# Build or load artifacts
# ---------------------------
def build_or_load():
    movies = pd.read_parquet(MOVIES_PATH) if MOVIES_PATH.exists() else None
    model = None
    if MODEL_PATH.exists() and MOVIES_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            movies = pd.read_parquet(MOVIES_PATH)
            print("Loaded cached model + movies.")
            return model, movies
        except Exception:
            pass  # rebuild if cache incompatible

    print("Preparing data & training model (first run may take several minutes)…")
    ratings, movies = load_ml25m_2015_2025()
    model = MFRecommender(n_components=N_COMPONENTS, random_state=42).fit(ratings)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Training complete; model cached.")
    return model, movies

model, movies = build_or_load()

# Precompute top choices for a fast dropdown
topN = 1500
top_movies = (movies.merge(
    pd.Series(name="cnt", index=[], dtype="int64"), how="left", left_on="item_id", right_index=True)
)
# If you want popularity sort, compute from ratings when building; here we just sort by title:
choices = movies.sort_values("title")["title"].tolist()

def recommend_ui(selected_titles: list, topk: int):
    if not selected_titles:
        return pd.DataFrame(columns=["title","year","score"])
    liked_ids = movies[movies["title"].isin(selected_titles)]["item_id"].tolist()
    recs = model.recommend_from_liked(liked_ids, topk=topk, exclude_item_ids=liked_ids)
    df = pd.DataFrame(recs, columns=["item_id","score"]).merge(
        movies, on="item_id", how="left")[["title","year","score"]]
    return df

demo = gr.Interface(
    fn=recommend_ui,
    inputs=[
        gr.Dropdown(choices=choices, multiselect=True, label="Pick a few movies you like (2015–2025)"),
        gr.Slider(5, 30, value=10, step=1, label="Top-K")
    ],
    outputs=gr.Dataframe(label="Recommendations"),
    title="🎬 Movie Recommender (2015–2025) — MovieLens-25M",
    description="Matrix Factorization (TruncatedSVD) trained on a modern subset of MovieLens-25M. First run downloads data and trains, then caches the model."
)

if __name__ == "__main__":
    # For local dev. In Spaces, the service manager starts the app automatically.
    demo.launch()

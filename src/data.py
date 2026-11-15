from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ROOT = Path(__file__).resolve().parents[1]
HOUR = 3_600_000


def load_data(data_dir=ROOT / "data/brut", sample=False):
    data_dir = Path(data_dir)
    files = [data_dir / "clicks_sample.csv"] if sample else sorted((data_dir / "clicks").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Clics absents : {data_dir}")
    columns = ["user_id", "click_article_id", "click_timestamp"]
    available = pd.read_csv(files[0], nrows=0).columns
    columns += [name for name in ("session_id", "session_start") if name in available]
    clicks = pd.concat([pd.read_csv(file, usecols=columns) for file in files], ignore_index=True)
    quality = {"raw_clicks": len(clicks), "files": len(files)}
    clean = clicks.dropna()
    quality["missing_rows"] = len(clicks) - len(clean)
    clicks = clean
    quality["outside_24h_session"] = 0
    if "session_start" in clicks:
        delay = clicks.click_timestamp - clicks.session_start
        valid = delay.between(0, 24 * HOUR)
        quality["outside_24h_session"] = int((~valid).sum())
        clicks = clicks.loc[valid]
    clicks = clicks[columns[:3]].astype({"user_id": "int32", "click_article_id": "int32", "click_timestamp": "int64"})
    quality["duplicates"] = int(clicks.duplicated().sum())
    clicks = clicks.drop_duplicates().sort_values(["user_id", "click_timestamp"], kind="stable").reset_index(drop=True)
    metadata = pd.read_csv(data_dir / "articles_metadata.csv")
    with (data_dir / "articles_embeddings.pickle").open("rb") as file:
        embeddings = np.asarray(pickle.load(file), dtype=np.float32)
    if len(metadata) != len(embeddings) or metadata.article_id.duplicated().any():
        raise ValueError("Les métadonnées et embeddings ne correspondent pas")
    required = ["article_id", "category_id", "created_at_ts"]
    if metadata[required].isna().any().any() or not np.isfinite(embeddings).all():
        raise ValueError("Métadonnées ou embeddings incomplets")
    if not clicks.click_article_id.isin(metadata.article_id).all():
        raise ValueError("Un article consulté manque dans le catalogue")
    published = clicks.click_article_id.map(metadata.set_index("article_id").created_at_ts)
    before_publication = clicks.click_timestamp < published
    quality["before_publication"] = int(before_publication.sum())
    clicks = clicks.loc[~before_publication].reset_index(drop=True)
    quality["retained_clicks"] = len(clicks)
    return clicks, metadata, embeddings, quality


def time_periods(clicks):
    end = int(clicks.click_timestamp.max()) + 1
    if end - int(clicks.click_timestamp.min()) >= 6 * 24 * HOUR:
        return {"validation_start": end - 96 * HOUR, "test_start": end - 48 * HOUR, "end": end}
    return {"validation_start": int(clicks.click_timestamp.quantile(.6)),
            "test_start": int(clicks.click_timestamp.quantile(.8)), "end": end}


def prepare_catalog(metadata, embeddings, clicks, cutoff, dimensions=64):
    fit = (metadata.created_at_ts < cutoff) & metadata.article_id.isin(
        clicks.loc[clicks.click_timestamp < cutoff, "click_article_id"].unique())
    rows = np.flatnonzero(fit.to_numpy())
    if len(rows) < 2:
        raise ValueError("Pas assez d'articles pour ajuster l'ACP")
    rows = np.random.default_rng(42).choice(rows, min(50000, len(rows)), replace=False)
    dimensions = min(dimensions, embeddings.shape[1], len(rows) - 1)
    pca = PCA(n_components=dimensions, svd_solver="randomized", random_state=42).fit(embeddings[rows])
    vectors = np.empty((len(metadata), dimensions), dtype=np.float32)
    for start in range(0, len(metadata), 50000):
        batch = pca.transform(embeddings[start:start + 50000]).astype(np.float32)
        batch /= np.maximum(np.linalg.norm(batch, axis=1, keepdims=True), 1e-12)
        vectors[start:start + len(batch)] = batch
    order = np.argsort(metadata.article_id.to_numpy())
    catalog = {"article_ids": metadata.article_id.to_numpy(dtype=np.int32)[order],
               "vectors": vectors[order], "categories": metadata.category_id.to_numpy(dtype=np.int32)[order],
               "published_at": metadata.created_at_ts.to_numpy(dtype=np.int64)[order],
               "pca_mean": pca.mean_.astype(np.float32), "pca_components": pca.components_.astype(np.float32)}
    return catalog, float(pca.explained_variance_ratio_.sum())

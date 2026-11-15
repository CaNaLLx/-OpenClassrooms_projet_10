import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.data import ROOT, load_data, prepare_catalog, time_periods
from src.recommender import Recommender


def collaborative_neighbors(history, article_ids, neighbors=50):
    recent = history.groupby("user_id", sort=False).tail(20)[["user_id", "click_article_id"]].drop_duplicates()
    users = np.sort(recent.user_id.unique())
    active_articles = np.sort(recent.click_article_id.unique())
    matrix = sparse.csr_matrix((np.ones(len(recent), dtype=np.float32),
                               (np.searchsorted(users, recent.user_id),
                                np.searchsorted(active_articles, recent.click_article_id))),
                              shape=(len(users), len(active_articles)))
    item_users = normalize(matrix.T.tocsr(), norm="l2", axis=1)
    user_items = item_users.T.tocsr()
    global_rows = np.searchsorted(article_ids, active_articles)
    counts = np.zeros(len(article_ids), dtype=np.int32)
    all_indices, all_scores = [], []
    for start in range(0, len(active_articles), 512):
        similarities = (item_users[start:start + 512] @ user_items).tocsr()
        for local in range(similarities.shape[0]):
            begin, end = similarities.indptr[local:local + 2]
            ids = similarities.indices[begin:end]
            values = similarities.data[begin:end]
            valid = (ids != start + local) & (values > 0)
            ids, values = ids[valid], values[valid]
            selected = np.lexsort((ids, -values))[:neighbors]
            all_indices.append(global_rows[ids[selected]].astype(np.int32))
            all_scores.append(values[selected])
            counts[global_rows[start + local]] = len(selected)
    return {"neighbor_offsets": np.r_[0, np.cumsum(counts)],
            "neighbor_indices": np.concatenate(all_indices).astype(np.int32),
            "neighbor_scores": np.concatenate(all_scores).astype(np.float32)}


def fit_models(clicks, catalog, cutoff=None):
    cutoff = int(clicks.click_timestamp.max()) + 1 if cutoff is None else int(cutoff)
    history = clicks.loc[clicks.click_timestamp < cutoff]
    if history.empty:
        raise ValueError("Aucun historique avant la coupure")
    user_ids = np.sort(history.user_id.unique()).astype(np.int32)
    article_ids = catalog["article_ids"]
    recent = history.groupby("user_id", sort=False).tail(10)
    user_positions = np.searchsorted(user_ids, recent.user_id)
    article_positions = np.searchsorted(article_ids, recent.click_article_id)
    ranks = recent.groupby("user_id", sort=False).cumcount(ascending=False).to_numpy()
    profiles = np.zeros((len(user_ids), catalog["vectors"].shape[1]), dtype=np.float32)
    np.add.at(profiles, user_positions, catalog["vectors"][article_positions] * np.power(.8, ranks[:, None]))
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-12)
    last_articles = np.full((len(user_ids), 10), -1, dtype=np.int32)
    last_articles[user_positions, ranks] = recent.click_article_id.to_numpy()
    seen = history[["user_id", "click_article_id"]].drop_duplicates().sort_values(["user_id", "click_article_id"])
    counts_per_user = np.bincount(np.searchsorted(user_ids, seen.user_id), minlength=len(user_ids))
    counts = np.bincount(np.searchsorted(article_ids, seen.click_article_id), minlength=len(article_ids)).astype(np.int32)
    last_times = history.groupby("user_id").click_timestamp.max().reindex(user_ids).to_numpy()
    arrays = {**catalog, "user_ids": user_ids, "profiles": profiles, "last_articles": last_articles,
              "last_times": last_times,
              "history_offsets": np.r_[0, np.cumsum(counts_per_user)],
              "history_articles": seen.click_article_id.to_numpy(dtype=np.int32), "counts": counts}
    start = time.perf_counter()
    arrays.update(collaborative_neighbors(history, article_ids))
    print(f"CF ajusté : {len(history):,} clics, {len(arrays['neighbor_indices']):,} voisins, {time.perf_counter()-start:.1f} s", flush=True)
    reference = max(int(history.click_timestamp.max()), int(catalog["published_at"].max()))
    return Recommender(arrays, cutoff, reference)


if __name__ == "__main__":
    from src.experiment import run_experiment
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/brut")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--max-cases", type=int, default=2000)
    parser.add_argument("--sample", action="store_true")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.output_dir, args.max_cases, args.sample)

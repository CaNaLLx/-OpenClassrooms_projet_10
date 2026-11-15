import time

import numpy as np
import pandas as pd

from src.data import HOUR
from src.recommender import METHODS
from src.train import fit_models


def evaluation_events(clicks, start, end, max_cases=2000, step_hours=6):
    if max_cases < 1:
        raise ValueError("Le nombre de cas doit être positif")
    first_user = clicks.groupby("user_id").click_timestamp.min()
    first_pair = clicks.groupby(["user_id", "click_article_id"]).click_timestamp.min()
    windows = []
    for snapshot in range(int(start), int(end), step_hours * HOUR):
        rows = clicks.loc[clicks.click_timestamp.between(snapshot, min(snapshot + step_hours * HOUR, end) - 1)]
        pairs = pd.MultiIndex.from_frame(rows[["user_id", "click_article_id"]])
        valid = (rows.user_id.map(first_user).to_numpy() < snapshot) & (first_pair.reindex(pairs).to_numpy() >= snapshot)
        rows = rows.loc[valid].drop_duplicates("user_id", keep="first")
        rows = rows[["user_id", "click_article_id", "click_timestamp"]].copy()
        rows["snapshot_time"] = snapshot
        windows.append(rows)
    events = pd.concat(windows, ignore_index=True)
    eligible = len(events)
    if not eligible:
        raise ValueError("Aucun cas évaluable dans cette période")
    if len(events) > max_cases:
        events = events.sample(max_cases, random_state=42)
    return events.sort_values(["snapshot_time", "user_id"]).reset_index(drop=True), eligible


def evaluate(clicks, catalog, events):
    details, catalog_union = [], set()
    for snapshot, cases in events.groupby("snapshot_time", sort=True):
        print(f"Fenêtre {pd.to_datetime(snapshot, unit='ms', utc=True)} : {len(cases)} cas", flush=True)
        model = fit_models(clicks, catalog, int(snapshot))
        d = model.data
        for user_id, target, timestamp in cases[["user_id", "click_article_id", "click_timestamp"]].itertuples(index=False, name=None):
            available = model.candidates(timestamp)
            catalog_union.update(d["article_ids"][available].tolist())
            target_row = model.article_positions([target])[0]
            target_new = bool(d["counts"][target_row] == 0)
            target_available = bool(target_row in available)
            for method in (*METHODS, "popular"):
                before = time.perf_counter()
                result = model.recommend(user_id, method, timestamp)
                latency = (time.perf_counter() - before) * 1000
                recs = result["article_ids"]
                rank = recs.index(target) + 1 if target in recs else 0
                positions = model.article_positions(recs)
                categories = d["categories"][positions]
                seeds = model.article_positions(result["history"])
                history_categories = d["categories"][seeds]
                values, frequencies = np.unique(history_categories, return_counts=True)
                dominant = values[np.argmax(frequencies)] if len(values) else -1
                vectors = d["vectors"][positions]
                similarity = vectors @ vectors.T
                concentration = np.unique(categories, return_counts=True)[1].max() / max(len(categories), 1)
                details.append({
                    "user_id": int(user_id), "target_id": int(target), "timestamp": int(timestamp),
                    "snapshot_time": int(snapshot), "method": method, "target_new": target_new,
                    "target_available": target_available, "hit": int(rank > 0),
                    "reciprocal_rank": 1 / rank if rank else 0.,
                    "ndcg": 1 / np.log2(rank + 1) if rank else 0.,
                    "article_ids": recs, "categories": categories.tolist(), "history": result["history"],
                    "fallback_share": result["fallback_count"] / 5,
                    "distinct_categories": len(np.unique(categories)), "category_concentration": concentration,
                    "dominant_history_share": float(np.mean(categories == dominant)),
                    "intra_list_similarity": float(similarity[np.triu_indices(len(recs), 1)].mean()) if len(recs) > 1 else 0.,
                    "new_article_share": float(np.mean(d["counts"][positions] == 0)),
                    "personalized_new_article_share": sum(a["training_readers"] == 0 and a["source"] == "personalized" for a in result["articles"]) / 5,
                    "mean_readers": float(d["counts"][positions].mean()),
                    "mean_age_hours": float(((timestamp - d["published_at"][positions]) / HOUR).mean()),
                    "latency_ms": latency,
                })
    frame = pd.DataFrame(details)
    metrics = {}
    for method, rows in frame.groupby("method", sort=False):
        recommended = set(article for recs in rows.article_ids for article in recs)
        grouped = {}
        for label, condition in (("new_article", rows.target_new), ("known_article", ~rows.target_new)):
            part = rows.loc[condition]
            grouped[label] = {"cases": len(part), "hit_rate_at_5": float(part.hit.mean()) if len(part) else None}
        metrics[method] = {
            "cases": len(rows), "hits": int(rows.hit.sum()), "hit_rate_at_5": float(rows.hit.mean()),
            "mrr_at_5": float(rows.reciprocal_rank.mean()), "ndcg_at_5": float(rows.ndcg.mean()),
            "catalog_coverage_at_5": len(recommended) / max(len(catalog_union), 1),
            **{name: float(rows[name].mean()) for name in (
                "fallback_share", "distinct_categories", "category_concentration", "dominant_history_share",
                "intra_list_similarity", "new_article_share", "personalized_new_article_share", "mean_readers", "mean_age_hours")},
            "median_latency_ms": float(rows.latency_ms.median()), "target_groups": grouped,
        }
    return {"cases": len(events), "unique_users": int(events.user_id.nunique()),
            "candidate_recall": float(frame.target_available.mean()),
            "candidate_catalog_size": len(catalog_union), "methods": metrics}, details


def choose_method(validation, require_new_articles=True):
    methods = validation["methods"]
    best_accuracy = max(METHODS, key=lambda name: (methods[name]["hit_rate_at_5"], methods[name]["mrr_at_5"]))
    if require_new_articles:
        eligible = [name for name in METHODS if methods[name]["personalized_new_article_share"] > 0]
    else:
        eligible = list(METHODS)
    if not eligible:
        raise ValueError("Aucune méthode ne satisfait la personnalisation des articles sans clics")
    selected = max(eligible, key=lambda name: (methods[name]["hit_rate_at_5"], methods[name]["mrr_at_5"]))
    return {"selected_method": selected, "best_validation_accuracy": best_accuracy,
            "require_personalized_new_articles": require_new_articles, "eligible_methods": eligible,
            "selection_data": "validation_only",
            "reason": "Personnaliser des articles sans clic est une exigence du MVP. Choix de la meilleure pertinence de validation parmi les méthodes qui le démontrent."
                      if require_new_articles else "Meilleure pertinence de validation, puis MRR en cas d'égalité."}

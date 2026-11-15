from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.data import ROOT, load_data, prepare_catalog, time_periods
from src.evaluate import choose_method, evaluate, evaluation_events
from src.recommender import METHODS
from src.train import fit_models


def comparative_examples(model, clicks, validation_details):
    frame = pd.DataFrame(validation_details)
    content = frame.loc[frame.method == "content"].copy()
    cf = frame.loc[frame.method == "collaborative"].copy()
    merged = content.merge(cf, on=["user_id", "snapshot_time"], suffixes=("_content", "_cf"))
    merged["concentration_gap"] = merged.category_concentration_content - merged.category_concentration_cf
    bubble = merged.sort_values(["concentration_gap", "category_concentration_content"], ascending=False).iloc[0]
    concentration = {
        "title": "Concentration des recommandations : un risque de bulle à observer",
        "user_id": int(bubble.user_id), "snapshot_time": int(bubble.snapshot_time),
        "history": bubble.history_content,
        "content_articles": bubble.article_ids_content, "collaborative_articles": bubble.article_ids_cf,
        "content_categories": bubble.categories_content, "collaborative_categories": bubble.categories_cf,
        "content_concentration": float(bubble.category_concentration_content),
        "collaborative_concentration": float(bubble.category_concentration_cf),
        "content_intra_similarity": float(bubble.intra_list_similarity_content),
        "collaborative_intra_similarity": float(bubble.intra_list_similarity_cf),
        "interpretation": "Exemple descriptif sélectionné pour illustrer une différence de concentration. Les moyennes globales sont fournies dans les métriques. Cela ne démontre pas un enfermement à long terme ni un effet causal de bulle.",
    }
    cold_candidates = content.loc[content.personalized_new_article_share > 0].sort_values(
        ["target_new", "hit", "personalized_new_article_share"], ascending=False)
    cold_article = None
    for row in cold_candidates.itertuples(index=False):
        observed = set(clicks.loc[clicks.click_timestamp < row.snapshot_time, "click_article_id"])
        other = cf.loc[(cf.user_id == row.user_id) & (cf.snapshot_time == row.snapshot_time)].iloc[0]
        new_ids = [article for article in row.article_ids if article not in observed and article not in other.article_ids]
        if new_ids:
            article = row.target_id if row.target_id in new_ids else new_ids[0]
            cold_article = {"title": "Article sans consultation dans l'apprentissage", "user_id": int(row.user_id),
                            "article_id": int(article), "training_clicks": 0, "snapshot_time": int(row.snapshot_time),
                            "history": row.history, "content_articles": row.article_ids,
                            "collaborative_articles": other.article_ids,
                            "content_rank": row.article_ids.index(article) + 1,
                            "is_actual_next_click": article == row.target_id,
                            "interpretation": "Le contenu peut classer cet article grâce à son embedding. Le CF ne possède pas de voisins appris pour lui. Sa présence dans les suggestions démontre sa prise en compte, pas la satisfaction de l'utilisateur."}
            break
    active_user = int(model.data["user_ids"][-1])
    history = [int(article) for article in model.data["last_articles"][-1] if article >= 0]
    unknown = int(model.data["user_ids"].max()) + 1
    new_user = {"title": "Nouvel utilisateur : sans historique puis avec une première préférence",
                "user_id": unknown, "seed_articles": history[:1],
                "without_history": {method: model.recommend(unknown, method) for method in METHODS},
                "with_first_preference": {method: model.recommend(unknown, method, seed_articles=history[:1]) for method in METHODS},
                "interpretation": "Sans historique ni préférence, les deux approches utilisent un repli commun. Une première préférence permet de calculer un profil de contenu. Le CF peut utiliser cette préférence si l'article possède déjà des voisins comportementaux."}
    positions = model.candidates(model.reference_time)
    return {"concentration": concentration, "new_article": cold_article, "new_user": new_user,
            "current_catalog": {"reference_time": model.reference_time, "cutoff": model.cutoff,
                                "candidate_articles": len(positions),
                                "articles_without_readers": int((model.data["counts"][positions] == 0).sum()),
                                "example_known_user": active_user}}


def save_results(output_dir, model, quality, periods, variance, validation, test, decision,
                 validation_count, test_count, details, scenarios, sample):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.selected_method = decision["selected_method"]
    model.save(output_dir / "recommender.npz")
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(), "sample_data": bool(sample),
        "data_quality": quality, "periods": periods, "protocol": "rolling_6h_validation_then_test",
        "hyperparameters": {"content_dimensions": model.data["vectors"].shape[1], "profile_reads": 10,
                            "cf_reads_per_user": 20, "cf_neighbors": 50, "candidate_days": [7, 30]},
        "pca_explained_variance": variance, "validation_eligible": validation_count, "test_eligible": test_count,
        "validation": validation, "test": test, "decision": decision,
        "users": len(model.data["user_ids"]), "catalog_articles": len(model.data["article_ids"]),
        "reference_time": model.reference_time, "training_cutoff": model.cutoff,
        "artifact_bytes": (output_dir / "recommender.npz").stat().st_size,
    }
    for name, value in (("metrics.json", metrics), ("scenarios.json", scenarios)):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "evaluation_cases.jsonl").open("w", encoding="utf-8") as stream:
        for row in details:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return metrics


def run_experiment(data_dir=ROOT / "data/brut", output_dir=ROOT / "models", max_cases=2000, sample=False):
    started = time.perf_counter()
    clicks, metadata, embeddings, quality = load_data(data_dir, sample)
    periods = time_periods(clicks)
    catalog, variance = prepare_catalog(metadata, embeddings, clicks, periods["validation_start"])
    events, validation_count = evaluation_events(clicks, periods["validation_start"], periods["test_start"], max_cases)
    validation, validation_details = evaluate(clicks, catalog, events)
    decision = choose_method(validation)
    print("Décision figée avant le test :", decision, flush=True)
    events, test_count = evaluation_events(clicks, periods["test_start"], periods["end"], max_cases)
    test, test_details = evaluate(clicks, catalog, events)
    model = fit_models(clicks, catalog)
    scenarios = comparative_examples(model, clicks, validation_details)
    metrics = save_results(output_dir, model, quality, periods, variance, validation, test, decision,
                           validation_count, test_count, validation_details + test_details, scenarios, sample)
    print(json.dumps({"decision": decision, "test": test, "seconds": time.perf_counter() - started}, indent=2), flush=True)
    return metrics

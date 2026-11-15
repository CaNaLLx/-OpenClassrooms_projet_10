import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recommender import Recommender

NAMES = {"content": "Basé sur le contenu", "collaborative": "Filtrage collaboratif"}
SOURCES = {"personalized": "Personnalisé", "popularity": "Repli : popularité", "freshness": "Repli : fraîcheur"}
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(ROOT / "models")))
MODEL_PATH = MODEL_DIR / "recommender.npz"

st.set_page_config(page_title="My Content", layout="wide")
st.title("My Content")
st.write("Cinq articles par utilisateur, avec deux approches à comparer.")


@st.cache_resource
def load_model(path, modified):
    return Recommender.load(path)


try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    scenarios = json.loads((MODEL_DIR / "scenarios.json").read_text(encoding="utf-8"))
except (OSError, ValueError, KeyError):
    st.error("Le modèle n'est pas disponible. Exécutez d'abord le notebook de comparaison.")
    st.stop()

d = model.data
now = pd.Timestamp.now(tz="UTC")
offset = now - pd.to_datetime(model.reference_time, unit="ms", utc=True)
st.caption("Simulation : la dernière création ou consultation du jeu de données correspond à aujourd'hui. Les écarts entre dates sont conservés.")

new_user = st.checkbox("Tester un nouvel utilisateur")
seeds = None
if new_user:
    user_id = int(d["user_ids"].max()) + 1
    recent = np.argsort(d["published_at"])[-30:][::-1]
    popular = np.argsort(d["counts"])[-30:][::-1]
    preferences = list(dict.fromkeys(d["article_ids"][np.r_[recent, popular]].tolist()))
    preference = st.selectbox("Premier article apprécié (facultatif)", [None, *preferences],
                              format_func=lambda x: "Aucune préférence" if x is None else str(x))
    seeds = [] if preference is None else [int(preference)]
    st.caption("Cette préférence est simulée pour le test ; aucun compte n'est créé.")
else:
    manual = st.checkbox("Saisir un identifiant")
    if manual:
        user_id = int(st.number_input("Identifiant utilisateur", min_value=0, value=0, step=1))
    else:
        positions = np.argsort(d["last_times"])[-1000:][::-1]
        user_id = int(st.selectbox("Utilisateur", d["user_ids"][positions].tolist()))

period = st.radio("Catalogue à utiliser", ["Dernière activité", "Fin des consultations"], horizontal=True)
at_time = model.reference_time if period == "Dernière activité" else model.cutoff - 1
display_time = pd.to_datetime(at_time, unit="ms", utc=True) + offset
st.caption(f"Date de simulation du catalogue : {display_time:%d/%m/%Y %H:%M} UTC.")
if period == "Dernière activité":
    st.caption("Le catalogue contient des publications postérieures aux dernières consultations. Le collaboratif peut donc utiliser le repli pour des articles sans clics.")

position = model.user_position(user_id)
history = seeds if seeds is not None else ([] if position is None else [int(x) for x in d["last_articles"][position] if x >= 0])
if history:
    st.write("Lectures ou préférences utilisées :", ", ".join(str(article) for article in history))
else:
    st.info("Sans historique ni préférence, les deux méthodes utilisent le même repli.")


def display_result(result):
    if len(result["article_ids"]) != 5:
        st.error("Le catalogue ne contient pas cinq articles disponibles et non lus.")
        return
    frame = pd.DataFrame(result["articles"])
    frame.insert(0, "Rang", range(1, 6))
    frame["published_at"] = (pd.to_datetime(frame.published_at, unit="ms", utc=True) + offset).dt.strftime("%d/%m/%Y %H:%M")
    frame["source"] = frame.source.map(SOURCES)
    frame["score"] = frame.score.round(4)
    frame = frame.rename(columns={"article_id": "Article", "category_id": "Catégorie", "published_at": "Publication UTC",
                                  "score": "Score", "training_readers": "Lecteurs dans l'apprentissage", "source": "Origine"})
    st.dataframe(frame, hide_index=True, width="stretch")
    st.caption(f"{5-result['fallback_count']}/5 suggestions personnalisées ; {result['fallback_count']}/5 issues du repli.")


recommend_tab, compare_tab, cases_tab, results_tab = st.tabs(
    ["Recommandations", "Comparer les approches", "Exemples testés", "Résultats"])

with recommend_tab:
    options = [model.selected_method, *[method for method in NAMES if method != model.selected_method]]
    method = st.selectbox("Approche", options, format_func=NAMES.get)
    st.caption(f"Approche retenue pour le besoin métier : {NAMES[model.selected_method]}.")
    if st.button("Recommander cinq articles", type="primary"):
        try:
            display_result(model.recommend(user_id, method, at_time, seeds))
        except ValueError as error:
            st.error(str(error))
    st.caption("Le jeu fournit les identifiants et catégories des articles, sans titre ni lien.")

with compare_tab:
    st.write("Même utilisateur, même historique, même date et mêmes articles disponibles.")
    if st.button("Comparer les deux approches"):
        columns = st.columns(2)
        for column, method in zip(columns, ("collaborative", "content")):
            with column:
                st.subheader(NAMES[method])
                try:
                    display_result(model.recommend(user_id, method, at_time, seeds))
                except ValueError as error:
                    st.error(str(error))
    st.caption("Les scores des deux méthodes ont des échelles différentes ; ils ne sont pas des probabilités.")

with cases_tab:
    concentration = scenarios["concentration"]
    st.subheader("1. Concentration des suggestions")
    st.write(f"Utilisateur {concentration['user_id']} — exemple extrait de la validation historique.")
    st.write("Lectures utilisées :", ", ".join(str(x) for x in concentration["history"]))
    table = pd.DataFrame({"Approche": ["Contenu", "Collaboratif"],
                          "Articles": [concentration["content_articles"], concentration["collaborative_articles"]],
                          "Catégories": [concentration["content_categories"], concentration["collaborative_categories"]],
                          "Part de la catégorie dominante": [concentration["content_concentration"], concentration["collaborative_concentration"]]})
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(concentration["interpretation"])
    st.subheader("2. Démarrage à froid d'un article")
    cold = scenarios.get("new_article")
    if cold:
        st.write(f"L'article {cold['article_id']} comptait zéro consultation avant la coupure. "
                 f"Le contenu le propose au rang {cold['content_rank']} pour l'utilisateur {cold['user_id']}.")
        if cold["is_actual_next_click"]:
            st.write("Cet article correspond à la prochaine lecture effectivement observée dans ce cas de validation.")
        st.dataframe(pd.DataFrame({"Contenu": cold["content_articles"], "Collaboratif": cold["collaborative_articles"]}), hide_index=True)
        st.caption(cold["interpretation"])
    else:
        st.info("Aucun exemple contrasté trouvé dans cette exécution ; consulter les résultats globaux.")
    st.subheader("3. Nouvel utilisateur")
    st.write("Utilisez la case « Tester un nouvel utilisateur » pour comparer le repli sans préférence puis la personnalisation après un premier choix.")

with results_tab:
    decision = metrics["decision"]
    st.subheader("Décision métier")
    st.write(decision["reason"])
    st.write("Meilleure pertinence de validation :", NAMES[decision["best_validation_accuracy"]])
    rows = {}
    for method in NAMES:
        values = metrics["test"]["methods"][method]
        rows[NAMES[method]] = {"Hit Rate@5 (%)": 100 * values["hit_rate_at_5"], "MRR@5": values["mrr_at_5"],
                               "Couverture (%)": 100 * values["catalog_coverage_at_5"],
                               "Catégories / 5": values["distinct_categories"],
                               "Articles sans clic personnalisés (%)": 100 * values["personalized_new_article_share"],
                               "Repli (%)": 100 * values["fallback_share"]}
    st.dataframe(pd.DataFrame(rows).T.round(3), width="stretch")
    st.caption(f"{metrics['test']['cases']} cas communs de test ; {metrics['test']['unique_users']} utilisateurs distincts. Décision prise sur validation avant le test.")
    st.caption("Ces mesures concernent le prochain clic historique. Elles ne constituent pas un taux de clic ni une mesure de satisfaction en production.")

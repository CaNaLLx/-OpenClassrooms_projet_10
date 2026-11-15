from pathlib import Path

import numpy as np


HOUR = 3_600_000
METHODS = ("collaborative", "content")
ARRAYS = ("article_ids", "vectors", "categories", "published_at", "pca_mean", "pca_components",
          "user_ids", "profiles", "history_offsets", "history_articles", "last_articles", "last_times", "counts",
          "neighbor_offsets", "neighbor_indices", "neighbor_scores")


def top_indices(scores, article_ids, number=5):
    number = min(number, len(scores))
    if number == 0:
        return np.array([], dtype=np.int64)
    threshold = np.partition(scores, len(scores) - number)[len(scores) - number]
    pool = np.flatnonzero(scores >= threshold)
    return pool[np.lexsort((article_ids[pool], -scores[pool]))[:number]]


class Recommender:
    def __init__(self, arrays, cutoff, reference_time, selected_method="content"):
        self.data = {name: np.asarray(arrays[name]) for name in ARRAYS}
        self.cutoff = int(cutoff)
        self.reference_time = int(reference_time)
        self.selected_method = selected_method

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.data, format_version=np.int32(1), cutoff=self.cutoff,
                            reference_time=self.reference_time, selected_method=self.selected_method)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as values:
            if int(values["format_version"]) != 1:
                raise ValueError("Format de modèle incompatible")
            return cls({name: values[name] for name in ARRAYS}, int(values["cutoff"]),
                       int(values["reference_time"]), str(values["selected_method"]))

    def user_position(self, user_id):
        ids = self.data["user_ids"]
        position = int(np.searchsorted(ids, user_id))
        return position if position < len(ids) and ids[position] == user_id else None

    def article_positions(self, article_ids):
        ids = self.data["article_ids"]
        values = np.asarray(article_ids, dtype=np.int64)
        positions = np.searchsorted(ids, values)
        valid = positions < len(ids)
        valid[valid] &= ids[positions[valid]] == values[valid]
        if not valid.all():
            raise ValueError("Identifiant d'article absent du catalogue")
        return positions

    def history(self, user_id):
        position = self.user_position(int(user_id))
        if position is None:
            return np.array([], dtype=np.int32)
        start, stop = self.data["history_offsets"][position:position + 2]
        return self.data["history_articles"][start:stop]

    def candidates(self, at_time):
        published = self.data["published_at"]
        available = published <= at_time
        for days in (7, 30):
            recent = np.flatnonzero(available & (published >= at_time - days * 24 * HOUR))
            if len(recent) >= 5:
                return recent
        return np.flatnonzero(available)

    def scores(self, user_id, method, at_time=None, seed_articles=None):
        if method not in (*METHODS, "popular"):
            raise ValueError("Méthode inconnue")
        timestamp = self.reference_time if at_time is None else int(at_time)
        if timestamp < self.cutoff - 1:
            raise ValueError("Cette date précède les données utilisées pour apprendre le modèle")
        d = self.data
        position = self.user_position(int(user_id))
        if seed_articles is None:
            seeds = [] if position is None else [int(x) for x in d["last_articles"][position] if x >= 0]
        else:
            seeds = list(dict.fromkeys(int(x) for x in seed_articles))[:10]
        seed_positions = self.article_positions(seeds)
        if np.any(d["published_at"][seed_positions] > timestamp):
            raise ValueError("Une préférence concerne un article pas encore publié")
        excluded = np.union1d(self.history(user_id), seeds)
        candidates = self.candidates(timestamp)
        candidates = candidates[~np.isin(d["article_ids"][candidates], excluded)]
        if len(candidates) < 5:
            candidates = np.flatnonzero((d["published_at"] <= timestamp) & ~np.isin(d["article_ids"], excluded))
        scores = np.zeros(len(candidates), dtype=np.float32)
        if method == "content" and len(seeds):
            if seed_articles is None and position is not None:
                profile = d["profiles"][position]
            else:
                weights = np.power(.8, np.arange(len(seeds)))
                profile = (d["vectors"][seed_positions] * weights[:, None]).sum(axis=0)
                profile /= max(float(np.linalg.norm(profile)), 1e-12)
            scores = d["vectors"][candidates] @ profile
        elif method == "collaborative":
            for row, weight in zip(seed_positions, np.power(.8, np.arange(len(seeds)))):
                start, stop = d["neighbor_offsets"][row:row + 2]
                targets = d["neighbor_indices"][start:stop]
                locations = np.searchsorted(candidates, targets)
                valid = locations < len(candidates)
                valid[valid] &= candidates[locations[valid]] == targets[valid]
                np.add.at(scores, locations[valid], weight * d["neighbor_scores"][start:stop][valid])
        elif method == "popular":
            scores = d["counts"][candidates].astype(np.float32)
        return candidates, scores, seeds

    def recommend(self, user_id, method=None, at_time=None, seed_articles=None, n=5):
        if int(user_id) < 0 or n < 1:
            raise ValueError("Identifiant ou nombre de recommandations invalide")
        method = self.selected_method if method is None else method
        timestamp = self.reference_time if at_time is None else int(at_time)
        candidates, scores, seeds = self.scores(user_id, method, timestamp, seed_articles)
        d = self.data
        personalized = (scores > 0) if method == "collaborative" else np.ones(len(scores), dtype=bool)
        if method == "popular" or not seeds:
            personalized[:] = False
        indices = np.flatnonzero(personalized)
        first = indices[top_indices(scores[indices], d["article_ids"][candidates[indices]], n)]
        sources = ["personalized"] * len(first)
        if len(first) < n:
            remaining = np.setdiff1d(np.arange(len(candidates)), first)
            order = np.lexsort((d["article_ids"][candidates[remaining]],
                                -d["published_at"][candidates[remaining]], -d["counts"][candidates[remaining]]))
            extra = remaining[order[:n - len(first)]]
            sources += ["popularity" if d["counts"][candidates[index]] > 0 else "freshness" for index in extra]
            first = np.r_[first, extra]
        articles = []
        for index, source in zip(first, sources):
            row = candidates[index]
            articles.append({"article_id": int(d["article_ids"][row]), "category_id": int(d["categories"][row]),
                             "published_at": int(d["published_at"][row]), "score": float(scores[index]),
                             "training_readers": int(d["counts"][row]), "source": source})
        return {"user_id": int(user_id), "method": method, "reference_time": timestamp,
                "article_ids": [article["article_id"] for article in articles], "articles": articles,
                "history": seeds, "fallback_count": sum(source != "personalized" for source in sources)}

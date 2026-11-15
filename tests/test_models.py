import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import HOUR, prepare_catalog
from src.recommender import Recommender
from src.train import fit_models


class ModelsTest(unittest.TestCase):
    def setUp(self):
        self.clicks = pd.DataFrame({"user_id": [10, 10, 20, 20, 30, 30, 40, 40, 99],
                                   "click_article_id": [1, 3, 1, 3, 2, 4, 2, 4, 1],
                                   "click_timestamp": np.arange(9) * 1000 + 100 * HOUR})
        self.catalog = {"article_ids": np.arange(1, 9, dtype=np.int32),
                        "vectors": np.array([[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float32),
                        "categories": np.array([1, 2, 2, 1, 1, 2, 3, 4]),
                        "published_at": np.zeros(8, dtype=np.int64),
                        "pca_mean": np.zeros(2), "pca_components": np.eye(2)}
        self.model = fit_models(self.clicks, self.catalog)

    def test_two_distinct_algorithms(self):
        cf = self.model.recommend(99, "collaborative")
        content = self.model.recommend(99, "content")
        self.assertEqual(cf["article_ids"][0], 3)
        self.assertEqual(content["article_ids"][0], 4)
        self.assertNotEqual(cf["article_ids"], content["article_ids"])
        for result in (cf, content):
            self.assertEqual(len(set(result["article_ids"])), 5)
            self.assertNotIn(1, result["article_ids"])

    def test_new_article_can_receive_only_content_personalization(self):
        candidates, cf, _ = self.model.scores(99, "collaborative")
        _, content, _ = self.model.scores(99, "content")
        index = int(np.flatnonzero(self.model.data["article_ids"][candidates] == 5)[0])
        self.assertEqual(self.model.data["counts"][4], 0)
        self.assertEqual(cf[index], 0)
        self.assertGreater(content[index], 0)
        self.assertIn(5, self.model.recommend(99, "content")["article_ids"])

    def test_new_user_without_and_with_preference(self):
        cf = self.model.recommend(1000, "collaborative")
        content = self.model.recommend(1000, "content")
        self.assertEqual(cf["article_ids"], content["article_ids"])
        self.assertEqual(content["fallback_count"], 5)
        self.assertEqual(self.model.recommend(1000, "content", seed_articles=[1])["article_ids"][0], 4)
        self.assertEqual(self.model.recommend(1000, "collaborative", seed_articles=[1])["article_ids"][0], 3)

    def test_future_publications_excluded_and_last_activity_is_today(self):
        catalog = {**self.catalog, "published_at": self.catalog["published_at"].copy()}
        catalog["published_at"][-1] = 200 * HOUR
        model = fit_models(self.clicks, catalog)
        self.assertEqual(model.reference_time, 200 * HOUR)
        at_time = int(self.clicks.click_timestamp.max())
        self.assertNotIn(8, model.recommend(99, "content", at_time=at_time)["article_ids"])
        self.assertIn(7, model.candidates(200 * HOUR))

    def test_training_and_pca_use_only_past(self):
        future = pd.DataFrame({"user_id": [99], "click_article_id": [5], "click_timestamp": [200 * HOUR]})
        clicks = pd.concat([self.clicks, future], ignore_index=True)
        model = fit_models(clicks, self.catalog, cutoff=150 * HOUR)
        self.assertEqual(model.data["counts"][4], 0)
        self.assertNotIn(5, model.history(99))
        metadata = pd.DataFrame({"article_id": np.arange(1, 9), "category_id": np.ones(8), "created_at_ts": np.zeros(8)})
        before, _ = prepare_catalog(metadata, self.catalog["vectors"], clicks, 150 * HOUR, dimensions=2)
        embeddings = self.catalog["vectors"].copy()
        embeddings[4] = [1000, -1000]
        after, _ = prepare_catalog(metadata, embeddings, clicks, 150 * HOUR, dimensions=2)
        np.testing.assert_array_equal(before["pca_mean"], after["pca_mean"])
        np.testing.assert_array_equal(before["pca_components"], after["pca_components"])

    def test_export_reload_preserves_both_rankings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            self.model.save(path)
            loaded = Recommender.load(path)
            for method in ("collaborative", "content"):
                self.assertEqual(self.model.recommend(99, method), loaded.recommend(99, method))


if __name__ == "__main__":
    unittest.main()

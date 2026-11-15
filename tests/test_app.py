import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests import test_models


ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def setUp(self):
        fixture = test_models.ModelsTest()
        fixture.setUp()
        self.model = fixture.model
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.model.save(root / "recommender.npz")
        values = {"hit_rate_at_5": .1, "mrr_at_5": .05, "catalog_coverage_at_5": .5,
                  "distinct_categories": 3., "personalized_new_article_share": .1, "fallback_share": .1}
        metrics = {"decision": {"reason": "Personnaliser les articles sans clic.", "best_validation_accuracy": "collaborative"},
                   "test": {"methods": {method: values for method in ("content", "collaborative")},
                            "cases": 100, "unique_users": 100}}
        scenarios = {"concentration": {"user_id": 99, "history": [1],
                                       "content_articles": [4, 5], "collaborative_articles": [3, 2],
                                       "content_categories": [1, 1], "collaborative_categories": [2, 1],
                                       "content_concentration": 1., "collaborative_concentration": .5,
                                       "interpretation": "Exemple descriptif."}, "new_article": None}
        for name, content in (("metrics.json", metrics), ("scenarios.json", scenarios)):
            (root / name).write_text(json.dumps(content), encoding="utf-8")
        self.environment = patch.dict(os.environ, {"MODEL_DIR": str(root)})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_recommend_and_compare(self):
        app = AppTest.from_file(ROOT / "app/app.py", default_timeout=30).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.dataframe[0].value.Article.tolist(), self.model.recommend(99, "content")["article_ids"])
        app.button[1].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.dataframe[0].value.Article.tolist(), self.model.recommend(99, "collaborative")["article_ids"])
        self.assertEqual(app.dataframe[1].value.Article.tolist(), self.model.recommend(99, "content")["article_ids"])

    def test_new_user_and_first_preference(self):
        app = AppTest.from_file(ROOT / "app/app.py", default_timeout=30).run()
        app.checkbox[0].set_value(True).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.dataframe[0].value.Origine.str.startswith("Repli").all())
        app.selectbox[0].set_value(1).run()
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.dataframe[0].value.Article.iloc[0], 4)
        self.assertEqual(app.dataframe[0].value.Origine.iloc[0], "Personnalisé")


if __name__ == "__main__":
    unittest.main()

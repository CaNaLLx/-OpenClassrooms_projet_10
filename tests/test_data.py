import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.data import HOUR, load_data


class DataTest(unittest.TestCase):
    def test_invalid_times_and_duplicates_are_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"user_id": [1, 1, 2, 3], "click_article_id": [1, 1, 2, 1],
                          "click_timestamp": [2000, 2000, 3000, 100 * HOUR],
                          "session_start": [1000] * 4}).to_csv(root / "clicks_sample.csv", index=False)
            pd.DataFrame({"article_id": [1, 2], "category_id": [1, 2],
                          "created_at_ts": [1000, 5000]}).to_csv(root / "articles_metadata.csv", index=False)
            with (root / "articles_embeddings.pickle").open("wb") as file:
                pickle.dump(np.eye(2, dtype=np.float32), file)
            clicks, _, _, quality = load_data(root, sample=True)
            self.assertEqual(clicks.user_id.tolist(), [1])
            self.assertEqual(quality["raw_clicks"], 4)
            self.assertEqual(quality["outside_24h_session"], 1)
            self.assertEqual(quality["duplicates"], 1)
            self.assertEqual(quality["before_publication"], 1)
            self.assertEqual(quality["retained_clicks"], 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch

import numpy as np

from src.data.splits import split_known_and_unknown
from src.evaluation.metrics import Method, l2_distance
from src.evaluation.open_set import _best_threshold, evaluate_open_set


class OpenSetTests(unittest.TestCase):
    def setUp(self):
        self.items = [
            (class_name, f"/{class_name}/{index}.ply")
            for class_name in ("a", "b", "c", "d")
            for index in range(20)
        ]

    def test_open_set_classes_never_leak_into_known_splits(self):
        train, val, test, unknown_val, unknown_test, known, unknown = (
            split_known_and_unknown(self.items, open_set_classes=2, seed=42)
        )

        self.assertTrue(set(known).isdisjoint(unknown))
        self.assertEqual({item[0] for item in train + val + test}, set(known))
        self.assertEqual(
            {item[0] for item in unknown_val + unknown_test}, set(unknown)
        )
        self.assertTrue(
            {item[0] for item in unknown_val}.isdisjoint(
                {item[0] for item in unknown_test}
            )
        )

    def test_open_set_class_selection_is_deterministic(self):
        first = split_known_and_unknown(
            self.items, open_set_classes=2, seed=7
        )[-1]
        second = split_known_and_unknown(
            self.items, open_set_classes=2, seed=7
        )[-1]

        self.assertEqual(first, second)

    def test_threshold_separates_known_and_unknown_scores(self):
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        targets = np.asarray([0, 0, 1, 1])

        threshold = _best_threshold(scores, targets)

        self.assertTrue(0.2 < threshold < 0.8)

    def test_open_set_evaluation_rejects_distant_embedding(self):
        calibration = [
            ("known", "/known-val.ply", np.asarray([0.1])),
            ("novel-val", "/novel-val.ply", np.asarray([10.0])),
        ]
        test = [
            ("known", "/known-test.ply", np.asarray([0.2])),
            ("novel-test", "/novel-test.ply", np.asarray([9.0])),
        ]
        method = Method("L2", l2_distance, False)

        with patch(
            "src.evaluation.open_set._embed_set", side_effect=[calibration, test]
        ):
            result = evaluate_open_set(
                model=None,
                reference_embeddings={"known": np.asarray([0.0])},
                known_val_set=[("known", "/known-val.ply")],
                unknown_val_set=[("novel-val", "/novel-val.ply")],
                known_test_set=[("known", "/known-test.ply")],
                unknown_test_set=[("novel-test", "/novel-test.ply")],
                methods={"L2": method},
                n_points=1,
                device="cpu",
            )

        self.assertEqual(result["L2"]["balanced_accuracy"], 1.0)
        self.assertEqual(result["L2"]["open_set_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

import os
import tempfile
import unittest

import numpy as np

from src.evaluation.ref_strategies import (
    STRATEGY_NAMES,
    build_references_median,
    build_references_trimmed_mean,
    coordinate_trimmed_mean,
    save_all_strategies,
)


class RobustReferenceStrategyTests(unittest.TestCase):
    def test_median_all_is_calculated_per_class_and_coordinate(self):
        class_to_embs = {
            "a": [
                np.asarray([1.0, 20.0]),
                np.asarray([3.0, 10.0]),
                np.asarray([100.0, 0.0]),
            ],
            "b": [
                np.asarray([-5.0, 2.0]),
                np.asarray([5.0, 4.0]),
            ],
        }

        references = build_references_median(class_to_embs)

        np.testing.assert_array_equal(
            references["a"], np.asarray([3.0, 10.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            references["b"], np.asarray([0.0, 3.0], dtype=np.float32)
        )

    def test_trimmed_mean_removes_each_coordinate_tail(self):
        embeddings = np.asarray(
            [
                [0.0, 100.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
                [7.0, 70.0],
                [8.0, 80.0],
                [100.0, 0.0],
            ],
            dtype=np.float32,
        )

        result = coordinate_trimmed_mean(embeddings, proportion_to_cut=0.10)

        np.testing.assert_array_equal(
            result, np.asarray([4.5, 45.0], dtype=np.float32)
        )

    def test_trimmed_mean_rejects_invalid_proportions(self):
        embeddings = np.ones((3, 2), dtype=np.float32)

        for invalid in (-0.01, 0.5, 1.0):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    coordinate_trimmed_mean(embeddings, invalid)

    def test_robust_strategies_are_built_from_the_same_input(self):
        class_to_embs = {
            "a": [np.asarray([float(value)]) for value in range(20)]
        }
        expected_trimmed = build_references_trimmed_mean(
            class_to_embs, proportion_to_cut=0.05
        )["a"]

        with tempfile.TemporaryDirectory() as temp_dir:
            saved = save_all_strategies(temp_dir, class_to_embs)

            self.assertEqual(set(saved), STRATEGY_NAMES)
            median_path = os.path.join(
                temp_dir, "reference_embeddings_median_all.npz"
            )
            trimmed_path = os.path.join(
                temp_dir, "reference_embeddings_trimmed_mean_05.npz"
            )
            with np.load(median_path, allow_pickle=False) as median_data:
                np.testing.assert_array_equal(
                    median_data["a"], np.asarray([9.5], dtype=np.float32)
                )
            with np.load(trimmed_path, allow_pickle=False) as trimmed_data:
                np.testing.assert_array_equal(trimmed_data["a"], expected_trimmed)


if __name__ == "__main__":
    unittest.main()

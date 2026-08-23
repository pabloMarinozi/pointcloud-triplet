import unittest

import numpy as np

from src.evaluation.metrics import Method, l2_distance
from src.evaluation.postprocessing import (
    KReciprocalConfig,
    PCAWhitening,
    WhiteningConfig,
    k_reciprocal_rerank,
    metrics_from_rankings,
    rank_samples_k_reciprocal,
    reciprocal_rank_fusion,
)


class PostprocessingTests(unittest.TestCase):
    def test_whitening_is_fit_on_supplied_train_matrix(self):
        train = np.asarray(
            [[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]]
        )
        transform = PCAWhitening(
            WhiteningConfig(n_components=1, shrinkage=0.0)
        ).fit(train)

        whitened_train = transform.transform(train)
        transformed_query = transform.transform(np.asarray([[3.0, 6.0]]))

        self.assertEqual(whitened_train.shape, (4, 1))
        self.assertAlmostEqual(float(whitened_train.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(whitened_train.var(ddof=1)), 1.0, places=6)
        self.assertGreater(float(transformed_query[0, 0]), 0.0)

    def test_k_reciprocal_returns_query_to_gallery_distances(self):
        distances = np.asarray(
            [
                [0.0, 0.1, 0.8, 0.9],
                [0.1, 0.0, 0.7, 0.8],
                [0.8, 0.7, 0.0, 0.1],
                [0.9, 0.8, 0.1, 0.0],
            ],
            dtype=np.float32,
        )

        result = k_reciprocal_rerank(
            distances,
            query_count=1,
            config=KReciprocalConfig(k1=2, k2=1, lambda_value=0.3),
        )

        self.assertEqual(result.shape, (1, 3))
        self.assertEqual(int(np.argmin(result[0])), 0)

    def test_k_reciprocal_supports_multiprototype_references(self):
        samples = [("a", "/a/query.ply", np.asarray([0.1], dtype=np.float32))]
        references = {
            "a": np.asarray([[0.0], [0.2]], dtype=np.float32),
            "b": np.asarray([[5.0], [6.0]], dtype=np.float32),
        }

        rankings = rank_samples_k_reciprocal(
            samples,
            references,
            Method("L2", l2_distance, False),
            KReciprocalConfig(k1=1, k2=1, lambda_value=0.5),
        )

        self.assertEqual(rankings, [["a", "b"]])

    def test_reciprocal_rank_fusion_combines_sources(self):
        rankings = reciprocal_rank_fusion(
            {
                "centroid": [["a", "b", "c"]],
                "multiprototype": [["b", "a", "c"]],
                "median": [["a", "c", "b"]],
            },
            constant=60,
        )

        self.assertEqual(rankings[0][0], "a")
        self.assertEqual(set(rankings[0]), {"a", "b", "c"})

    def test_metrics_from_rankings_uses_one_based_true_rank(self):
        samples = [
            ("a", "/a.ply", np.asarray([0.0])),
            ("b", "/b.ply", np.asarray([1.0])),
        ]

        metrics = metrics_from_rankings(samples, [["a", "b"], ["a", "b"]])

        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["top5_accuracy"], 1.0)
        self.assertEqual(metrics["mrr"], 0.75)


if __name__ == "__main__":
    unittest.main()

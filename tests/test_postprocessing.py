import unittest

import numpy as np

from src.evaluation.metrics import Method, l2_distance
from src.evaluation.postprocessing import (
    KReciprocalConfig,
    PCAWhitening,
    WhiteningConfig,
    evaluate_postprocessing,
    k_reciprocal_rerank,
    metrics_from_rankings,
    rank_samples_k_reciprocal,
    reciprocal_rank_fusion,
    references_from_samples,
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

    def test_smoke_full_postprocessing_pipeline(self):
        train = [
            ("a", "/a/1.ply", np.asarray([0.0, 0.0], dtype=np.float32)),
            ("a", "/a/2.ply", np.asarray([0.2, 0.1], dtype=np.float32)),
            ("b", "/b/1.ply", np.asarray([5.0, 0.0], dtype=np.float32)),
            ("b", "/b/2.ply", np.asarray([5.2, 0.1], dtype=np.float32)),
            ("c", "/c/1.ply", np.asarray([0.0, 5.0], dtype=np.float32)),
            ("c", "/c/2.ply", np.asarray([0.1, 5.2], dtype=np.float32)),
        ]
        val = [
            ("a", "/a/val.ply", np.asarray([0.1, 0.1], dtype=np.float32)),
            ("b", "/b/val.ply", np.asarray([5.1, 0.1], dtype=np.float32)),
            ("c", "/c/val.ply", np.asarray([0.1, 5.1], dtype=np.float32)),
        ]
        test = [
            ("a", "/a/test.ply", np.asarray([0.05, 0.0], dtype=np.float32)),
            ("b", "/b/test.ply", np.asarray([5.0, 0.05], dtype=np.float32)),
            ("c", "/c/test.ply", np.asarray([0.0, 5.05], dtype=np.float32)),
        ]
        strategy_names = ["centroid_all", "median_all"]
        references = references_from_samples(train, strategy_names, seed=42)

        report = evaluate_postprocessing(
            train_samples=train,
            val_samples=val,
            test_samples=test,
            references_by_strategy=references,
            methods={"L2 Distance": Method("L2 Distance", l2_distance, False)},
            whitening_configs=[WhiteningConfig(None, 1e-4)],
            reranking_configs=[KReciprocalConfig(2, 1, 0.5)],
            rrf_constants=[20],
            seed=42,
        )

        self.assertEqual(
            set(report["test"]),
            {"baseline", "whitening", "k_reciprocal", "fusion", "all"},
        )
        self.assertEqual(report["test"]["all"]["metrics"]["accuracy"], 1.0)
        self.assertEqual(
            report["protocol"],
            "select_hyperparameters_on_val_then_apply_unchanged_to_test",
        )


if __name__ == "__main__":
    unittest.main()

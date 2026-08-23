import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.evaluation.embed import (
    aggregate_view_embeddings,
    derive_sample_seed,
    embed_point_cloud_paths,
    embed_point_cloud_views,
    preprocess_point_cloud_path,
)
from src.evaluation.embedding_cache import (
    cache_paths,
    ensure_embedding_cache,
    sha256_file,
)
from src.evaluation.ref_strategies import coordinate_median
from src.evaluation.metrics import Method, l1_distance
from src.evaluation.report import evaluate_run_on_val


class MeanEmbeddingModel(torch.nn.Module):
    def embed(self, values):
        return values.mean(dim=2)


class EmbeddingCacheTests(unittest.TestCase):
    def setUp(self):
        self.points = np.arange(90, dtype=np.float32).reshape(30, 3)

    def test_coordinate_median_is_calculated_per_dimension(self):
        embeddings = np.asarray([[1, 20], [3, 10], [100, 0]], dtype=np.float32)

        result = coordinate_median(embeddings)

        np.testing.assert_array_equal(result, np.asarray([3, 10], dtype=np.float32))

    @patch("src.evaluation.embed.read_points_from_ply")
    def test_same_preprocessing_is_used_for_reference_and_query(self, read_points):
        read_points.return_value = self.points

        reference = preprocess_point_cloud_path(
            "/class/cloud.ply", 8, sampling="random", seed=7
        )
        query = preprocess_point_cloud_path(
            "/class/cloud.ply", 8, sampling="random", seed=7
        )

        np.testing.assert_array_equal(reference, query)

    @patch("src.evaluation.embed.read_points_from_ply")
    def test_embeddings_do_not_depend_on_traversal_order(self, read_points):
        read_points.return_value = self.points
        samples = [("a", "/a/one.ply"), ("b", "/b/two.ply")]
        model = MeanEmbeddingModel().eval()

        first = embed_point_cloud_paths(
            model, samples, 8, torch.device("cpu"), seed=11, batch_size=2,
            show_progress_every=0,
        )
        reversed_result = embed_point_cloud_paths(
            model, list(reversed(samples)), 8, torch.device("cpu"), seed=11,
            batch_size=2, show_progress_every=0,
        )

        first_by_path = {path: embedding for _, path, embedding in first}
        reversed_by_path = {path: embedding for _, path, embedding in reversed_result}
        for path in first_by_path:
            np.testing.assert_array_equal(first_by_path[path], reversed_by_path[path])

    def test_seed_changes_by_path_and_view(self):
        seed = derive_sample_seed(42, "/a/cloud.ply", 0)

        self.assertEqual(seed, derive_sample_seed(42, "/a/cloud.ply", 0))
        self.assertNotEqual(seed, derive_sample_seed(42, "/b/cloud.ply", 0))
        self.assertNotEqual(seed, derive_sample_seed(42, "/a/cloud.ply", 1))

    def test_multiview_aggregation_is_coordinate_wise(self):
        views = [
            ("a", "/a/cloud.ply", 2, np.asarray([100.0, 0.0])),
            ("a", "/a/cloud.ply", 0, np.asarray([1.0, 20.0])),
            ("a", "/a/cloud.ply", 1, np.asarray([3.0, 10.0])),
        ]

        median = aggregate_view_embeddings(views, "coordinate_median")
        mean = aggregate_view_embeddings(views, "coordinate_mean")

        np.testing.assert_array_equal(
            median[0][2], np.asarray([3.0, 10.0], dtype=np.float32)
        )
        np.testing.assert_allclose(
            mean[0][2], np.asarray([104.0 / 3.0, 10.0], dtype=np.float32)
        )

    @patch("src.evaluation.embed.read_points_from_ply")
    def test_multiview_reads_each_cloud_once(self, read_points):
        read_points.return_value = self.points
        samples = [("a", "/a/one.ply"), ("b", "/b/two.ply")]

        result = embed_point_cloud_views(
            MeanEmbeddingModel().eval(),
            samples,
            8,
            torch.device("cpu"),
            seed=11,
            view_ids=range(4),
            batch_size=8,
            show_progress_every=0,
        )

        self.assertEqual(read_points.call_count, 2)
        self.assertEqual(len(result), 8)
        self.assertEqual([item[2] for item in result[:4]], [0, 1, 2, 3])

    def test_fixed_embeddings_produce_identical_metrics(self):
        embeddings = [
            ("a", "/a/one.ply", np.asarray([0.0], dtype=np.float32)),
            ("b", "/b/two.ply", np.asarray([1.0], dtype=np.float32)),
        ]
        arguments = dict(
            model=None,
            reference_embeddings={
                "a": np.asarray([0.0], dtype=np.float32),
                "b": np.asarray([1.0], dtype=np.float32),
            },
            val_set=[("a", "/a/one.ply"), ("b", "/b/two.ply")],
            methods={"L1": Method("L1", l1_distance, False)},
            n_points=8,
            device="cpu",
            export_csv=False,
            precomputed_embeddings=embeddings,
        )

        first = evaluate_run_on_val(**arguments)
        second = evaluate_run_on_val(**arguments)

        self.assertEqual(first, second)

    @patch("src.evaluation.embedding_cache.embed_point_cloud_views")
    def test_incompatible_manifest_regenerates_cache(self, embed_views):
        samples = [("a", "/a/one.ply")]
        embed_views.return_value = [
            ("a", "/a/one.ply", 0, np.asarray([1.0, 2.0], dtype=np.float32))
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "model.pt")
            with open(checkpoint, "wb") as file:
                file.write(b"checkpoint")
            checkpoint_hash = sha256_file(checkpoint)
            common = dict(
                cache_dir=temp_dir,
                split="train",
                model=None,
                samples=samples,
                n_points=8,
                device="cpu",
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_hash,
                seed=42,
                batch_size=2,
            )

            ensure_embedding_cache(**common, sampling="random")
            ensure_embedding_cache(**common, sampling="random")
            ensure_embedding_cache(**common, sampling="fps")

            self.assertEqual(embed_views.call_count, 2)
            _, manifest_path = cache_paths(temp_dir, "train")
            with open(manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(manifest["sampling"], "fps")

    @patch("src.evaluation.embedding_cache.embed_point_cloud_views")
    def test_cache_contains_required_metadata(self, embed_views):
        samples = [("person", "/person/person_VID_20200101_120000_nube_1.ply")]
        embed_views.return_value = [
            (samples[0][0], samples[0][1], 0, np.asarray([0.5], dtype=np.float32))
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "model.pt")
            with open(checkpoint, "wb") as file:
                file.write(b"checkpoint")
            ensure_embedding_cache(
                cache_dir=temp_dir,
                split="val",
                model=None,
                samples=samples,
                n_points=8,
                device="cpu",
                checkpoint_path=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                sampling="random",
                seed=42,
                video_index={"VID_20200101_120000": "frontal"},
            )

            cache_path, _ = cache_paths(temp_dir, "val")
            with np.load(cache_path, allow_pickle=False) as data:
                self.assertEqual(
                    set(data.files),
                    {"path", "label", "video", "capture_form", "view_id", "seed", "embedding"},
                )
                self.assertEqual(str(data["capture_form"][0]), "frontal")

    @patch("src.evaluation.embedding_cache.embed_point_cloud_views")
    def test_cache_stores_all_views_and_returns_one_embedding_per_cloud(
        self, embed_views
    ):
        samples = [("a", "/a/one.ply")]
        embed_views.return_value = [
            ("a", "/a/one.ply", view_id, np.asarray([float(view_id)]))
            for view_id in range(4)
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "model.pt")
            with open(checkpoint, "wb") as file:
                file.write(b"checkpoint")

            result = ensure_embedding_cache(
                cache_dir=temp_dir,
                split="train",
                model=None,
                samples=samples,
                n_points=8,
                device="cpu",
                checkpoint_path=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                views=4,
                view_aggregation="coordinate_median",
            )

            self.assertEqual(len(result), 1)
            np.testing.assert_array_equal(
                result[0][2], np.asarray([1.5], dtype=np.float32)
            )
            cache_path, _ = cache_paths(temp_dir, "train")
            with np.load(cache_path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["view_id"], np.arange(4))

    def test_evaluation_records_latency_without_changing_metrics(self):
        runtime = {}
        results = evaluate_run_on_val(
            model=None,
            reference_embeddings={"a": np.asarray([0.0], dtype=np.float32)},
            val_set=[("a", "/a/one.ply")],
            methods={"L1": Method("L1", l1_distance, False)},
            n_points=8,
            device="cpu",
            export_csv=False,
            precomputed_embeddings=[
                ("a", "/a/one.ply", np.asarray([0.0], dtype=np.float32))
            ],
            runtime_stats=runtime,
        )

        self.assertEqual(results["L1"]["accuracy"], 1.0)
        self.assertEqual(runtime["L1"]["query_count"], 1)
        self.assertGreaterEqual(runtime["L1"]["latency_ms_per_query"], 0.0)
        self.assertGreater(runtime["L1"]["peak_process_rss_mb"], 0.0)


if __name__ == "__main__":
    unittest.main()

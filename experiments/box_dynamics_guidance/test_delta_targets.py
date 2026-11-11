"""Delta-target invariants and durable exports; no simulator or training run."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    from . import delta_targets
    from .dataset import FEATURE_SPEC, INPUT_DIM, DatasetError, LoadedDataset, WindowSplit, file_sha256
    from .train_smoke import _metrics, _normalization
except ImportError:
    import delta_targets
    from dataset import FEATURE_SPEC, INPUT_DIM, DatasetError, LoadedDataset, WindowSplit, file_sha256
    from train_smoke import _metrics, _normalization


def synthetic_split(name, count=7):
    random = np.random.RandomState(31 if name == "train" else 71)
    persistence = random.normal(size=(count, 9)).astype(np.float32)
    persistence[:, 2:5] += [4.0, -3.0, 0.7]
    future = persistence + random.normal(scale=0.03, size=(count, 9)).astype(np.float32)
    return WindowSplit(
        x=random.normal(size=(count, INPUT_DIM)).astype(np.float32),
        y=future, persistence=persistence,
        episode_ids=np.asarray([name + "_episode"] * count),
        window_start_steps=np.arange(1260, 1260 + count * 20, 20, dtype=np.int64),
    )


class DeltaTargetTests(unittest.TestCase):
    def setUp(self):
        self.split = synthetic_split("train")
        self.selected = delta_targets.DELTA_INDICES
        self.unchanged = np.asarray([1, 5, 6, 7, 8])

    def test_roundtrip_and_unchanged_components_do_not_mutate_sources(self):
        future, persistence = self.split.y.copy(), self.split.persistence.copy()
        target = delta_targets.make_targets(self.split)
        np.testing.assert_array_equal(target[:, self.selected],
                                      future[:, self.selected] - persistence[:, self.selected])
        np.testing.assert_array_equal(target[:, self.unchanged], future[:, self.unchanged])
        decoded = delta_targets.decode_targets(target, self.split.persistence)
        np.testing.assert_allclose(decoded, future, rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(decoded[:, self.unchanged], future[:, self.unchanged])
        self.assertFalse(np.shares_memory(target, self.split.y))
        self.assertFalse(np.shares_memory(decoded, target))
        np.testing.assert_array_equal(self.split.y, future)
        np.testing.assert_array_equal(self.split.persistence, persistence)

    def test_normalization_reuses_absolute_scales_and_zero_predicts_persistence(self):
        center, scale, absolute_mean = delta_targets.target_normalization(self.split.y)
        expected_mean, expected_scale = _normalization(self.split.y)
        np.testing.assert_array_equal(scale, expected_scale)
        np.testing.assert_array_equal(absolute_mean, expected_mean)
        np.testing.assert_array_equal(center[self.selected], 0)
        np.testing.assert_array_equal(center[self.unchanged], expected_mean[self.unchanged])
        normalized_output = np.zeros_like(self.split.y)
        absolute = delta_targets.decode_targets(normalized_output * scale + center,
                                                 self.split.persistence)
        np.testing.assert_array_equal(absolute[:, self.selected], self.split.persistence[:, self.selected])
        np.testing.assert_array_equal(absolute[:, self.unchanged],
                                      np.broadcast_to(expected_mean[self.unchanged],
                                                      (len(absolute), len(self.unchanged))))

    def test_delta_loss_equals_original_physical_weighted_loss(self):
        center, scale, _ = delta_targets.target_normalization(self.split.y)
        target = delta_targets.make_targets(self.split).astype(np.float64)
        normalized_target = (target - center) / scale
        normalized_prediction = np.random.RandomState(91).normal(size=target.shape)
        delta_loss = np.mean((normalized_prediction - normalized_target) ** 2)
        absolute_prediction = delta_targets.decode_targets(
            normalized_prediction * scale + center, self.split.persistence)
        original_loss = _metrics(absolute_prediction, self.split.y, scale)["normalized_mse"]
        self.assertAlmostEqual(delta_loss, original_loss, places=7)

    def test_feature_spec_is_independent_of_original(self):
        original = deepcopy(FEATURE_SPEC)
        copied = delta_targets.DELTA_FEATURE_SPEC
        self.assertEqual(copied["target_representation"], delta_targets.TARGET_SPEC)
        self.assertIsNot(copied["inputs"], FEATURE_SPEC["inputs"])
        self.assertIsNot(copied["inputs"][0], FEATURE_SPEC["inputs"][0])
        self.assertNotIn("target_representation", FEATURE_SPEC)
        self.assertEqual(FEATURE_SPEC, original)

    def test_invalid_shapes_and_nonfinite_values_are_rejected(self):
        with self.assertRaises(DatasetError):
            delta_targets.decode_targets(self.split.y, self.split.persistence[:1])
        with self.assertRaises(DatasetError):
            delta_targets.target_normalization(np.empty((0, 9), np.float32))
        invalid = self.split.y.copy()
        invalid[0, 0] = np.nan
        with self.assertRaises(DatasetError):
            delta_targets.decode_targets(invalid, self.split.persistence)


class DeltaArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="box_delta_targets_test_")
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        (source / "dataset.json").write_text('{"fixture": "source manifest"}\n', encoding="utf-8")
        (source / "plan.json").write_text('{"fixture": "source plan"}\n', encoding="utf-8")
        self.dataset = LoadedDataset(
            root=source,
            splits={name: synthetic_split(name) for name in ("train", "val")},
            metadata={
                "dataset_path": str(source),
                "dataset_manifest_sha256": file_sha256(source / "dataset.json"),
                "plan_sha256": file_sha256(source / "plan.json"),
            },
        )
        self.output = self.root / "targets"

    def tearDown(self):
        self.temporary.cleanup()

    def test_archives_load_without_pickle_and_source_is_unchanged(self):
        source_files = {path: file_sha256(path) for path in self.dataset.root.iterdir()}
        source_splits = deepcopy(self.dataset.splits)
        source_metadata = deepcopy(self.dataset.metadata)
        manifest = delta_targets.save_targets(self.dataset, self.output)
        self.assertEqual(json.loads((self.output / "manifest.json").read_text()), manifest)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["dataset_identity"], source_metadata)
        self.assertFalse((self.output / ".writing").exists())
        for name, split in self.dataset.splits.items():
            entry = manifest["splits"][name]
            archive = self.output / entry["path"]
            self.assertEqual(entry["sha256"], file_sha256(archive))
            self.assertEqual(entry["window_count"], len(split.y))
            self.assertEqual(entry["episode_count"], 1)
            with np.load(archive, allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), {"target", "absolute_target", "persistence",
                                                    "episode_ids", "window_start_steps"})
                for key in saved.files:
                    self.assertNotEqual(saved[key].dtype.kind, "O")
                np.testing.assert_array_equal(saved["target"], delta_targets.make_targets(split))
                for key, expected in (("absolute_target", split.y), ("persistence", split.persistence),
                                      ("episode_ids", split.episode_ids),
                                      ("window_start_steps", split.window_start_steps)):
                    np.testing.assert_array_equal(saved[key], expected)
            for field in ("x", "y", "persistence", "episode_ids", "window_start_steps"):
                np.testing.assert_array_equal(getattr(split, field), getattr(source_splits[name], field))
        self.assertEqual(source_files, {path: file_sha256(path) for path in self.dataset.root.iterdir()})
        self.assertEqual(self.dataset.metadata, source_metadata)
        module_directory = Path(delta_targets.__file__).resolve().parent
        for name, digest in manifest["sources"].items():
            self.assertEqual(digest, file_sha256(module_directory / name))

    def test_nonempty_output_is_preserved(self):
        self.output.mkdir()
        existing = self.output / "keep.txt"
        existing.write_text("keep this result", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            delta_targets.save_targets(self.dataset, self.output)
        self.assertEqual(existing.read_text(), "keep this result")
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_empty_output_accepted_and_reexport_refused(self):
        self.output.mkdir()
        delta_targets.save_targets(self.dataset, self.output)
        hashes = {path: file_sha256(path) for path in self.output.iterdir()}
        with self.assertRaises(FileExistsError):
            delta_targets.save_targets(self.dataset, self.output)
        self.assertEqual(hashes, {path: file_sha256(path) for path in self.output.iterdir()})

    def test_writing_inside_source_dataset_is_rejected(self):
        destination = self.dataset.root / "targets"
        with self.assertRaises(ValueError):
            delta_targets.save_targets(self.dataset, destination)
        self.assertFalse(destination.exists())

    def test_interrupted_export_does_not_publish_completion(self):
        with patch.object(delta_targets, "_write_npz", side_effect=OSError("simulated interrupted write")):
            with self.assertRaisesRegex(OSError, "interrupted write"):
                delta_targets.save_targets(self.dataset, self.output)
        self.assertFalse((self.output / "manifest.json").exists())
        with self.assertRaises(FileExistsError):
            delta_targets.save_targets(self.dataset, self.output)


if __name__ == "__main__":
    unittest.main()

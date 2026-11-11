"""Delta training contracts on temporary synthetic data; no simulator required.

Run: python -m unittest discover -s experiments/box_dynamics_guidance
     -p test_train_delta.py -v
"""

from contextlib import redirect_stderr
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import test_dataset as fixtures
import train_delta as training
import train_model as absolute_training
from dataset import FEATURE_SPEC, load_dataset
from guidance import Selector


class DeltaTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DatasetContractTests()
        self.fixture.setUp()
        self.dataset_root = self.fixture.root
        self.output_directory = tempfile.TemporaryDirectory(prefix="box_delta_training_test_")
        self.root = Path(self.output_directory.name)
        self.options = {
            "dataset_path": self.dataset_root, "device": "cpu", "hidden_dim": 16,
            "layers": 1, "batch_size": 5, "patience": 0, "progress": False,
        }

    def tearDown(self):
        self.output_directory.cleanup()
        self.fixture.tearDown()

    def run_training(self, name, **overrides):
        options = dict(self.options, output_path=self.root / name)
        options.update(overrides)
        with redirect_stderr(io.StringIO()):
            return training.train_model(**options)

    def checkpoint(self, name, filename="last.pt"):
        return torch.load(self.root / name / filename, map_location="cpu", weights_only=True)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_partial_epoch_interruption_resume_matches_continuous_training(self):
        expected_report = self.run_training("continuous", epochs=3)
        original_predict = training._predict
        calls = 0

        def interrupt_second_validation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # Weights have changed during epoch two, but it is not committed.
                raise KeyboardInterrupt()
            return original_predict(*args, **kwargs)

        with patch.object(training, "_predict", side_effect=interrupt_second_validation):
            with self.assertRaises(KeyboardInterrupt):
                self.run_training("interrupted", epochs=3)
        status = json.loads((self.root / "interrupted/status.json").read_text())
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["completed_epoch"], 1)
        self.assertEqual(self.checkpoint("interrupted")["completed_epoch"], 1)
        target_manifest = self.root / "interrupted/targets/manifest.json"
        target_hash = training.file_sha256(target_manifest)

        report = self.run_training("interrupted", epochs=3, resume=True)
        expected, resumed = self.checkpoint("continuous"), self.checkpoint("interrupted")
        for field in ("model_state_dict", "optimizer_state_dict", "rng_state", "normalization",
                      "best_model_state_dict", "best_epoch", "best_val_normalized_mse",
                      "early_stopping_anchor", "stale_epochs"):
            self.assert_nested_equal(expected[field], resumed[field])
        self.assertEqual(expected_report["val_model"], report["val_model"])
        self.assertEqual(len(expected["history"]), len(resumed["history"]))
        for first, second in zip(expected["history"], resumed["history"]):
            self.assertEqual({k: v for k, v in first.items() if k != "epoch_seconds"},
                             {k: v for k, v in second.items() if k != "epoch_seconds"})
        self.assertEqual(training.file_sha256(target_manifest), target_hash)

    def test_best_reload_reconstructs_absolute_validation_metrics(self):
        report = self.run_training("reload", epochs=3)
        saved = self.checkpoint("reload", "best.pt")
        dataset = load_dataset(self.dataset_root)
        val = dataset.splits["val"]
        config = saved["model_config"]
        model = absolute_training.ResponseMLP(config["hidden_dim"], config["layers"])
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        norm = {key: value.numpy() for key, value in saved["normalization"].items()}
        normalized = (val.x - norm["x_mean"]) / norm["x_std"]
        pieces = []
        with torch.no_grad():
            for start in range(0, len(normalized), self.options["batch_size"]):
                pieces.append(model(torch.from_numpy(normalized[start:start + 5])).numpy())
        # Independently decode instead of reusing the trainer's delta helper.
        prediction = np.concatenate(pieces) * norm["y_std"] + norm["y_mean"]
        prediction[:, 0] += val.persistence[:, 0]
        prediction[:, 2:5] += val.persistence[:, 2:5]
        actual = absolute_training._metrics(prediction, val.y, norm["y_std"])
        self.assertEqual(actual, report["val_model"])
        self.assertEqual(actual, saved["best_val_metrics"])
        self.assertEqual(actual["normalized_mse"], report["best_val_normalized_mse"])
        self.assertEqual(saved["best_epoch"], min(
            self.checkpoint("reload")["history"], key=lambda row: row["val_normalized_mse"])["epoch"])

    def test_input_and_loss_scales_match_absolute_training_with_zero_delta_center(self):
        absolute_feature_spec = copy.deepcopy(FEATURE_SPEC)
        self.run_training("delta_norm", epochs=1)
        with redirect_stderr(io.StringIO()):
            absolute_training.train_model(
                **dict(self.options, output_path=self.root / "absolute_norm", epochs=1))
        delta = self.checkpoint("delta_norm")["normalization"]
        absolute = self.checkpoint("absolute_norm")["normalization"]
        for field in ("x_mean", "x_std", "y_std"):
            self.assertTrue(torch.equal(delta[field], absolute[field]), field)
        self.assertTrue(torch.equal(delta["absolute_y_mean"], absolute["y_mean"]))
        self.assertTrue(torch.equal(delta["y_mean"][[0, 2, 3, 4]], torch.zeros(4)))
        self.assertTrue(torch.equal(delta["y_mean"][[1, 5, 6, 7, 8]],
                                    absolute["y_mean"][[1, 5, 6, 7, 8]]))
        # Fixture validation features have a deliberately different distribution.
        dataset = load_dataset(self.dataset_root)
        expected_x_mean = dataset.splits["train"].x.astype(np.float64).mean(axis=0).astype(np.float32)
        np.testing.assert_array_equal(delta["x_mean"].numpy(), expected_x_mean)
        self.assertFalse(np.allclose(expected_x_mean, np.concatenate([
            dataset.splits["train"].x, dataset.splits["val"].x]).mean(axis=0)))
        self.assertEqual(FEATURE_SPEC, absolute_feature_spec)

    def test_saved_teachers_change_only_angle_and_position_and_preserve_source(self):
        before = {path: training.file_sha256(path) for path in (
            self.dataset_root / "plan.json", self.dataset_root / "dataset.json",
            self.dataset_root / "trajectories/synthetic_train.npz", self.dataset_root / "trajectories/synthetic_val.npz")}
        self.run_training("teachers", epochs=1)
        dataset = load_dataset(self.dataset_root)
        manifest_path = self.root / "teachers/targets/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(set(manifest["splits"]), {"train", "val"})
        self.assertTrue(manifest["complete"])
        for name in ("train", "val"):
            source = dataset.splits[name]
            expected = source.y.copy()
            expected[:, 0] -= source.persistence[:, 0]
            expected[:, 2:5] -= source.persistence[:, 2:5]
            archive = manifest_path.parent / manifest["splits"][name]["path"]
            self.assertEqual(training.file_sha256(archive), manifest["splits"][name]["sha256"])
            with np.load(archive, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["target"], expected)
                np.testing.assert_array_equal(saved["absolute_target"], source.y)
                np.testing.assert_array_equal(saved["persistence"], source.persistence)
                np.testing.assert_array_equal(saved["episode_ids"], source.episode_ids)
                np.testing.assert_array_equal(saved["window_start_steps"], source.window_start_steps)
        self.assertEqual({path: training.file_sha256(path) for path in before}, before)

    def test_absolute_guidance_rejects_delta_artifact_and_delta_feature_spec(self):
        self.run_training("guard", epochs=1)
        saved = self.checkpoint("guard", "best.pt")
        self.assertEqual(saved["artifact_type"], "box_response_delta_best_predictor")
        self.assertNotEqual(saved["feature_spec"], FEATURE_SPEC)
        self.assertEqual(saved["feature_spec"]["inputs"], FEATURE_SPEC["inputs"])
        with self.assertRaisesRegex(ValueError, "matching trained box-response"):
            Selector(self.root / "guard/best.pt", "cpu")
        # The feature guard still protects against an accidentally reused type.
        saved["artifact_type"] = "box_response_best_predictor"
        renamed = self.root / "guard/wrong_artifact_type.pt"
        torch.save(saved, renamed)
        with self.assertRaisesRegex(ValueError, "matching trained box-response"):
            Selector(renamed, "cpu")

    def test_resume_refuses_changed_archive_without_touching_checkpoint(self):
        self.run_training("changed_bytes", epochs=1)
        original = training.file_sha256(self.root / "changed_bytes/last.pt")
        manifest_path = self.root / "changed_bytes/targets/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        path = manifest_path.parent / manifest["splits"]["train"]["path"]
        with path.open("ab") as stream:
            stream.write(b"tampered archive")
        with self.assertRaisesRegex(ValueError, "archive path/hash mismatch"):
            self.run_training("changed_bytes", epochs=2, resume=True)
        self.assertEqual(training.file_sha256(self.root / "changed_bytes/last.pt"), original)

    def test_resume_refuses_changed_teacher_even_with_updated_archive_hash(self):
        self.run_training("changed_teacher", epochs=1)
        original = training.file_sha256(self.root / "changed_teacher/last.pt")
        manifest_path = self.root / "changed_teacher/targets/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        path = manifest_path.parent / manifest["splits"]["train"]["path"]
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays["target"][0, 0] += .25
        np.savez_compressed(path, **arrays)
        manifest["splits"]["train"]["sha256"] = training.file_sha256(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "targets do not match"):
            self.run_training("changed_teacher", epochs=2, resume=True)
        self.assertEqual(training.file_sha256(self.root / "changed_teacher/last.pt"), original)

    def test_strict_best_and_early_stop_can_be_repaired_without_more_updates(self):
        original_metrics = training._metrics
        calls = 0

        def controlled_scores(*args, **kwargs):
            nonlocal calls
            value = original_metrics(*args, **kwargs)
            value["normalized_mse"] = (1., .98, .97)[min(calls, 2)]
            calls += 1
            return value

        options = {"epochs": 6, "patience": 2, "min_delta": .1}
        with patch.object(training, "_metrics", side_effect=controlled_scores):
            report = self.run_training("early", **options)
            self.assertEqual(report["stop_reason"], "early_stopping")
            self.assertEqual(report["completed_epochs"], 3)
            self.assertEqual(report["best_epoch"], 3)
            self.assertEqual(report["best_val_normalized_mse"], .97)
            last = self.checkpoint("early")
            self.assertEqual(last["early_stopping_anchor"], 1.)
            self.assertEqual(last["stale_epochs"], 2)
            previous_hash = training.file_sha256(self.root / "early/last.pt")
            for filename in ("best.pt", "history.jsonl", "report.json"):
                (self.root / "early" / filename).unlink()
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Unexpected optimization")):
                repaired = self.run_training("early", resume=True, **options)
            self.assertEqual(repaired["stop_reason"], "early_stopping")
            self.assertEqual(training.file_sha256(self.root / "early/last.pt"), previous_hash)
            self.assertEqual(self.checkpoint("early", "best.pt")["best_epoch"], 3)
            self.assertEqual(len((self.root / "early/history.jsonl").read_text().splitlines()), 3)

    def test_final_epoch_commit_can_be_finalized_without_more_updates(self):
        original_publish = training._publish_epoch

        def fail_after_last_commit(output, checkpoint):
            if checkpoint["completed_epoch"] == 2:
                raise RuntimeError("Synthetic failure after last.pt commit")
            original_publish(output, checkpoint)

        with patch.object(training, "_publish_epoch", side_effect=fail_after_last_commit):
            with self.assertRaisesRegex(RuntimeError, "after last.pt commit"):
                self.run_training("finalize", epochs=2)
        self.assertEqual(self.checkpoint("finalize")["completed_epoch"], 2)
        previous_hash = training.file_sha256(self.root / "finalize/last.pt")
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Unexpected optimization")):
            report = self.run_training("finalize", epochs=2, resume=True)
        self.assertEqual(report["completed_epochs"], 2)
        self.assertEqual(report["stop_reason"], "max_epochs")
        self.assertEqual(training.file_sha256(self.root / "finalize/last.pt"), previous_hash)
        self.assert_nested_equal(self.checkpoint("finalize", "best.pt")["model_state_dict"],
                                 self.checkpoint("finalize")["best_model_state_dict"])
        self.assertEqual(len((self.root / "finalize/history.jsonl").read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()

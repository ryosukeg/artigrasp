"""CPU-only synthetic training/resume tests; no real dataset or simulator used.

Run from this directory: python -m unittest -v test_train_model
"""

from contextlib import redirect_stderr
import io
import json
import unittest
from unittest.mock import patch

import torch

import test_dataset as fixtures
import train_model as training


class TrainingLifecycleTests(unittest.TestCase):
    def setUp(self):
        # Reuse the full hashed dataset contract fixture without importing its
        # TestCase class into this module's unittest discovery namespace.
        self.fixture = fixtures.DatasetContractTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.options = {
            "dataset_path": self.root, "device": "cpu", "hidden_dim": 16,
            "layers": 1, "batch_size": 5, "patience": 0, "progress": False,
        }

    def tearDown(self):
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

    def test_partial_epoch_interruption_resume_matches_uninterrupted(self):
        expected_report = self.run_training("continuous", epochs=3)
        original_predict = training._predict
        calls = 0

        def interrupt_second_validation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                # Epoch two has updated weights, but has not committed last.pt.
                raise KeyboardInterrupt()
            return original_predict(*args, **kwargs)

        with patch.object(training, "_predict", side_effect=interrupt_second_validation):
            with self.assertRaises(KeyboardInterrupt):
                self.run_training("interrupted", epochs=3)
        status = json.loads((self.root / "interrupted/status.json").read_text())
        self.assertEqual(status["status"], "interrupted")
        self.assertEqual(status["completed_epoch"], 1)
        self.assertEqual(self.checkpoint("interrupted")["completed_epoch"], 1)

        resumed_report = self.run_training("interrupted", epochs=3, resume=True)
        uninterrupted = self.checkpoint("continuous")
        resumed = self.checkpoint("interrupted")
        for field in ("model_state_dict", "optimizer_state_dict", "rng_state", "normalization",
                      "best_model_state_dict", "best_epoch", "best_val_normalized_mse", "stale_epochs"):
            self.assert_nested_equal(uninterrupted[field], resumed[field])
        self.assertEqual(expected_report["val_model"], resumed_report["val_model"])
        # Wall times differ, but optimization and validation values must not.
        for expected, actual in zip(uninterrupted["history"], resumed["history"]):
            self.assertEqual({k: v for k, v in expected.items() if k != "epoch_seconds"},
                             {k: v for k, v in actual.items() if k != "epoch_seconds"})

    def test_strict_best_selection_and_early_stop_finalization(self):
        original_metrics = training._metrics
        calls = 0

        def controlled_validation_scores(*args, **kwargs):
            nonlocal calls
            result = original_metrics(*args, **kwargs)
            # Isolate stopping/selection semantics from a fixture's learnability:
            # every epoch improves, but by less than min_delta=0.1.
            result["normalized_mse"] = (1.0, .98, .97)[min(calls, 2)]
            calls += 1
            return result

        options = {"epochs": 6, "patience": 2, "min_delta": .1}
        with patch.object(training, "_metrics", side_effect=controlled_validation_scores):
            report = self.run_training("early", **options)
            self.assertEqual(report["stop_reason"], "early_stopping")
            self.assertEqual(report["completed_epochs"], 3)
            self.assertEqual(report["best_epoch"], 3)
            self.assertEqual(report["best_val_normalized_mse"], .97)
            last = self.checkpoint("early")
            best = self.checkpoint("early", "best.pt")
            self.assertEqual(last["early_stopping_anchor"], 1.0)
            self.assertEqual(last["stale_epochs"], 2)
            self.assert_nested_equal(last["model_state_dict"], best["model_state_dict"])

            previous_hash = training.file_sha256(self.root / "early/last.pt")
            for filename in ("best.pt", "history.jsonl", "report.json"):
                (self.root / "early" / filename).unlink()
            with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Unexpected extra training")):
                repaired = self.run_training("early", resume=True, **options)
            self.assertEqual(repaired["completed_epochs"], 3)
            self.assertEqual(repaired["stop_reason"], "early_stopping")
            self.assertEqual(training.file_sha256(self.root / "early/last.pt"), previous_hash)
            self.assertEqual(self.checkpoint("early", "best.pt")["best_epoch"], 3)
            self.assertEqual(len((self.root / "early/history.jsonl").read_text().splitlines()), 3)

    def test_final_epoch_commit_can_be_finalized_without_more_training(self):
        original_publish = training._publish_epoch

        def interrupt_after_final_commit(output, checkpoint):
            if checkpoint["completed_epoch"] == 2:
                raise RuntimeError("Synthetic failure after last.pt commit")
            original_publish(output, checkpoint)

        with patch.object(training, "_publish_epoch", side_effect=interrupt_after_final_commit):
            with self.assertRaisesRegex(RuntimeError, "after last.pt commit"):
                self.run_training("finalize", epochs=2)
        status = json.loads((self.root / "finalize/status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertEqual(status["completed_epoch"], 2)
        self.assertFalse((self.root / "finalize/report.json").exists())
        previous_hash = training.file_sha256(self.root / "finalize/last.pt")

        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Unexpected extra training")):
            report = self.run_training("finalize", epochs=2, resume=True)
        self.assertEqual(report["completed_epochs"], 2)
        self.assertEqual(report["stop_reason"], "max_epochs")
        self.assertEqual(training.file_sha256(self.root / "finalize/last.pt"), previous_hash)
        last = self.checkpoint("finalize")
        best = self.checkpoint("finalize", "best.pt")
        self.assert_nested_equal(best["model_state_dict"], last["best_model_state_dict"])
        self.assertEqual(len((self.root / "finalize/history.jsonl").read_text().splitlines()), 2)
        self.assertEqual(json.loads((self.root / "finalize/status.json").read_text())["status"], "completed")


if __name__ == "__main__":
    unittest.main()

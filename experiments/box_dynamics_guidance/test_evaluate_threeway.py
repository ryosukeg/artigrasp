"""Matched scheduling and resume-contract tests without a simulator."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_threeway as evaluation


class ThreewayExecutionTests(unittest.TestCase):
    def test_every_method_receives_identical_batch_order_and_size(self):
        episodes = [{"episode_id": profile + str(i), "profile": profile}
                    for profile in ("normal", "ood_articulation") for i in range(5)]
        plan = {"config": {"runtime": {"batch_size": 3}},
                "profiles": ["normal", "ood_articulation"], "episodes": episodes}
        before = copy.deepcopy(plan)
        tasks = evaluation.tasks(plan)
        self.assertEqual(len(tasks), 12)
        seen = {method: [] for method in evaluation.METHODS}
        for offset in range(0, len(tasks), 3):
            chunk = tasks[offset:offset + 3]
            self.assertEqual([row[1] for row in chunk], list(evaluation.METHODS))
            self.assertEqual(chunk[0][2], chunk[1][2])
            self.assertEqual(chunk[0][2], chunk[2][2])
            for _, method, specs in chunk:
                seen[method].extend(row["episode_id"] for row in specs)
        for ids in seen.values():
            self.assertEqual(ids, [row["episode_id"] for row in episodes])
        self.assertEqual(plan, before)

    def test_loaders_receive_the_correct_frozen_predictor(self):
        plan = {"predictors": {"guided_absolute": {"path": "absolute.pt"},
                               "guided_delta": {"path": "delta.pt"}}}
        with patch.object(evaluation, "Selector") as absolute, patch.object(evaluation, "DeltaSelector") as delta:
            self.assertIsNone(evaluation.selector_for("baseline", plan, "cpu"))
            absolute.assert_not_called()
            delta.assert_not_called()
            self.assertIs(evaluation.selector_for("guided_absolute", plan, "cpu"), absolute.return_value)
            self.assertIs(evaluation.selector_for("guided_delta", plan, "cpu"), delta.return_value)
            absolute.assert_called_once_with("absolute.pt", "cpu")
            delta.assert_called_once_with("delta.pt", "cpu")
            with self.assertRaisesRegex(ValueError, "Unknown evaluation method"):
                evaluation.selector_for("unsupported", plan, "cpu")

    def test_plan_refuses_changed_candidates_even_with_consistent_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            plan = {"methods": list(evaluation.METHODS), "guidance": copy.deepcopy(evaluation.SETTINGS)}
            plan["guidance"]["candidates"][1][0] = 2.0
            evaluation.write_json(output / "plan.json", plan)
            evaluation.write_json(output / "provenance.json", {
                "plan_sha256": evaluation.file_sha256(output / "plan.json")})
            with patch.object(evaluation, "fingerprint", return_value={}):
                with self.assertRaisesRegex(ValueError, "frozen three-way"):
                    evaluation.check_plan(output)

    def test_resume_refuses_reordered_or_missing_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            evaluation.write_json(output / "plan.json", {"placeholder": True})
            specs = [{"episode_id": "first"}, {"episode_id": "second"}]
            task = ("batch", "guided_delta", specs)
            for rows in (specs[::-1], specs[:1], [specs[0], specs[0]]):
                evaluation.write_json(output / "batches/batch.json", {
                    "complete": True,
                    "plan_sha256": evaluation.file_sha256(output / "plan.json"),
                    "episodes": rows})
                with self.assertRaisesRegex(ValueError, "Batch IDs"):
                    evaluation.check_batch(output, task)


if __name__ == "__main__":
    unittest.main()

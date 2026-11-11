"""Report contracts: failure denominators, paired differences and exact controls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


_spec = importlib.util.spec_from_file_location(
    "box_evaluate_report_under_test", Path(__file__).with_name("evaluate_report.py"))
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def row(episode, method, angle_error, valid=True, target=.3):
    return {"episode_id": episode, "sample_index": int(episode), "method": method,
            "profile": "ood_articulation", "target_articulation_rad": target,
            "valid": valid, "paper_success": angle_error is not None and angle_error < .5,
            "ood_goal_success": angle_error is not None and angle_error < .5,
            "paper_articulation_error_rad": angle_error,
            "paper_position_error_m": .01, "paper_orientation_error_rad": .02,
            "opening_entry": {"position_error_m": .01, "orientation_error_rad": .02,
                              "articulation_rad": .8, "prior_lift_success": True},
            "guidance_candidate_counts": [3, 2, 2, 2, 2, 2, 2]}


class StatisticsTests(unittest.TestCase):
    def test_invalid_trials_are_failures_and_excluded_from_error_estimates(self):
        rows = [row("0", "guided", .1), row("1", "guided", .2, valid=False),
                row("2", "guided", None)]
        result = report.summarize_rows(rows)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["success"]["paper_success"], {"count": 1, "n": 3, "rate": 1 / 3})
        self.assertEqual(result["success"]["angle_lt_0p1"]["count"], 0)
        self.assertEqual(result["success"]["base_stable"]["count"], 2)
        error = result["errors"]["paper_articulation_error_rad"]
        self.assertEqual(error["mean"], .1)
        self.assertEqual(error["n"], 1)
        self.assertEqual(error["excluded_or_missing"], 2)
        self.assertEqual(error["missing_or_nonfinite_raw"], 1)
        self.assertEqual(result["guidance_candidates"]["counts"], [9, 6, 6, 6, 6, 6, 6])

    def test_paired_gains_losses_and_ci_use_within_episode_differences(self):
        pairs = [(row("0", "baseline", .8), row("0", "guided", .2)),
                 (row("1", "baseline", .1), row("1", "guided", .7)),
                 (row("2", "baseline", .4), row("2", "guided", .1, valid=False))]
        result = report.paired_summary(pairs)
        self.assertEqual(result["paper_success"]["gains"], 1)
        self.assertEqual(result["paper_success"]["losses"], 2)
        self.assertEqual(result["paper_success"]["rate_difference_guided_minus_baseline"], -1 / 3)
        error = result["angle_difference_guided_minus_baseline_rad"]
        self.assertEqual(error["n"], 2)
        self.assertEqual(error["excluded_or_missing_pairs"], 1)
        self.assertAlmostEqual(error["mean"], 0)
        np.testing.assert_allclose(error["bootstrap_95_percent_ci"], [-.6, .6])
        self.assertTrue(error["small_sample_warning"])
        self.assertEqual(result, report.paired_summary(pairs))

    def test_single_pair_does_not_report_a_false_zero_width_ci(self):
        result = report.paired_summary([(row("0", "baseline", .8), row("0", "guided", .2))])
        self.assertIsNone(result["angle_difference_guided_minus_baseline_rad"]["bootstrap_95_percent_ci"])

    def test_opening_tolerance_uses_the_actual_entry_angle_and_strict_goal_is_separate(self):
        first = row("0", "guided", .05, target=.3)
        first["initial_articulation_error_rad"] = .8
        first["articulation_start_error_rad"] = .2
        first["opening_entry"]["articulation_rad"] = .35
        first["strict_goal_success"] = False  # E.g. the earlier lift was not achieved.
        result = report.summarize_rows([first])
        counts = result["angle_tolerance_before_final"]
        self.assertEqual(counts["initial"]["angle_lt_0p5"]["count"], 0)
        self.assertEqual(counts["opening_before_phase_reset"]["angle_lt_0p5"]["count"], 1)
        self.assertEqual(counts["opening_entry"]["angle_lt_0p1"]["count"], 1)
        self.assertEqual(result["success"]["angle_lt_0p1"]["count"], 1)
        self.assertEqual(result["success"]["strict_goal_success"]["count"], 0)


class SavedReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.rows = []
        self.plan = {"config": {"seed": 202}, "episodes": [
            {"profile": "ood_articulation", "episode_id": "0"},
            {"profile": "ood_articulation", "episode_id": "1"}]}
        (self.output / "plan.json").write_text(json.dumps(self.plan))
        for episode, target in (("0", .3), ("1", 2.0)):
            for method in report.METHODS:
                entry = row(episode, method, .8 if method == "baseline" else .2, target=target)
                path = self.output / (episode + "_" + method + ".npz")
                goal = np.zeros(8, dtype=np.float32)
                goal[7] = target
                arrays = {"states": np.zeros((1561, 138), np.float32),
                          "actions_r": np.zeros((1560, 51), np.float32),
                          "actions_l": np.zeros((1560, 51), np.float32),
                          "final_states": np.zeros((301, 138), np.float32), "goal": goal}
                # The final response may differ; only the shared prefix is controlled.
                if method == "guided":
                    arrays["states"][1261:, 129] = .9
                    arrays["actions_r"][1260:, 0] = 1
                    arrays["final_states"][1:, 129] = .9
                np.savez_compressed(path, **arrays)
                entry.update(trajectory_path=path.name, trajectory_sha256=report._sha256(path))
                self.rows.append(entry)
        self.save_results()

    def save_results(self):
        self.manifest = {"complete": True, "plan_sha256": report._sha256(self.output / "plan.json"),
                         "episodes": self.rows}
        (self.output / "results.json").write_text(json.dumps(self.manifest))

    def mutate_array(self, key, position):
        entry = self.rows[1]
        path = self.output / entry["trajectory_path"]
        with np.load(path) as data:
            arrays = {name: data[name] for name in data.files}
        arrays[key][position] += 1
        np.savez_compressed(path, **arrays)
        entry["trajectory_sha256"] = report._sha256(path)
        self.save_results()

    def test_matching_prefix_allows_comparison_and_splits_ood_bands(self):
        result = report.write_report(self.output)
        self.assertTrue(result["comparison_conclusion_allowed"])
        self.assertEqual(result["pairing_audit"]["matching_pairs"], 2)
        self.assertEqual(result["groups"]["ood_small"]["guided"]["n"], 1)
        self.assertEqual(result["groups"]["ood_large"]["guided"]["n"], 1)
        self.assertEqual(result["groups"]["ood_articulation"]["paired"]["paper_success"]["gains"], 2)
        self.assertEqual(json.loads((self.output / "summary.json").read_text()), result)
        text = (self.output / "COMPARISON_REVIEW.md").read_text()
        self.assertIn("探索的な単一 seed", text)
        self.assertIn("2/2 (100.0%)", text)

    def test_phase_reset_state_mismatch_withholds_conclusion(self):
        self.mutate_array("final_states", (0, 13))
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_conclusion_allowed"])
        self.assertTrue(any("final_states" in issue for issue in result["pairing_audit"]["issues"]))
        self.assertIn("結論を保留", (self.output / "COMPARISON_REVIEW.md").read_text())

    def test_last_prefix_state_is_included_in_audit(self):
        self.mutate_array("states", (1260, 122))
        result = report.write_report(self.output)
        self.assertFalse(result["pairing_audit"]["passed"])

    def test_preintervention_action_mismatch_withholds_conclusion(self):
        self.mutate_array("actions_l", (1259, 0))
        result = report.write_report(self.output)
        self.assertFalse(result["pairing_audit"]["passed"])

    def test_corrupt_hash_is_reported_without_accepting_comparison(self):
        self.rows[1]["trajectory_sha256"] = "0" * 64
        self.save_results()
        result = report.write_report(self.output)
        self.assertFalse(result["pairing_audit"]["passed"])
        self.assertTrue(any("SHA256" in issue for issue in result["pairing_audit"]["issues"]))

    def test_missing_and_duplicate_results_cannot_pass(self):
        self.rows[1] = self.rows[0].copy()
        self.save_results()
        result = report.write_report(self.output)
        self.assertFalse(result["pairing_audit"]["passed"])
        issues = result["pairing_audit"]["issues"]
        self.assertTrue(any("Missing result" in issue for issue in issues))
        self.assertTrue(any("Duplicate result" in issue for issue in issues))

    def test_train_smoke_is_not_described_as_an_official_test_comparison(self):
        self.plan.update(split="train", purpose="technical_smoke")
        (self.output / "plan.json").write_text(json.dumps(self.plan))
        self.save_results()
        result = report.write_report(self.output)
        self.assertTrue(result["pairing_audit"]["passed"])
        self.assertFalse(result["comparison_conclusion_allowed"])
        text = (self.output / "COMPARISON_REVIEW.md").read_text()
        self.assertIn("公式 train 参照", text)
        self.assertIn("OOD 改善に関する結論は出しません", text)
        self.assertNotIn("公式 test 参照を使用", text)


if __name__ == "__main__":
    unittest.main()

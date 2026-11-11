"""Synthetic three-way reporting tests; no simulator or predictor import."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_threeway as report


class ThreewayReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.plan = {"schema_version": 1, "seed": 202, "split": "test",
                     "purpose": "exploratory_single_seed_paired_comparison",
                     "methods": list(report.METHODS), "predictors": {}, "episodes": [],
                     "guidance": {"horizon_steps": 20, "reselect_every_steps": 20,
                                  "phase_start_step": 1260, "candidates": report.CANDIDATES,
                                  "angle_scale_rad": .5, "position_scale_m": .05,
                                  "orientation_scale_rad": .2, "residual_squared_weight": .01}}
        for method in report.METHODS[1:]:
            directory = self.output / "models" / method
            directory.mkdir(parents=True)
            model = directory / "best.pt"
            model.write_bytes(method.encode())
            training = directory / "report.json"
            training.write_text(json.dumps({"artifact_sha256": {"best.pt": report._sha256(model)}}))
            self.plan["predictors"][method] = {"path": str(model), "sha256": report._sha256(model),
                                                 "training_report_sha256": report._sha256(training)}
        self.rows = []
        for index, (profile, angle) in enumerate((("normal", .55), ("ood_articulation", .3), ("ood_articulation", 2.0))):
            spec = {"episode_id": "box__{}__{:04d}".format(profile, index), "profile": profile,
                    "object": "box", "sample_index": index, "seed": 100 + index, "split": "test",
                    "grasp_label": {"seq_num": 9, "seq_name": "box_grab_01", "init_frame": 10},
                    "arti_label": {"seq_num": 9, "seq_name": "box_use_01", "init_frame": 20},
                    "target": {"articulation_rad": angle, "articulation_rule": "absolute", "lift_m": .15}}
            self.plan["episodes"].append(spec)
            for method, error in zip(report.METHODS, (.7, .2, .05)):
                goal = np.zeros(8, dtype=np.float32)
                goal[3], goal[7] = 1, angle
                states = np.zeros((1561, 138), dtype=np.float32)
                states[:, 125], states[:, 129] = 1, .8
                states[:, 130:138] = goal
                states[261:561, 124] = .15
                states[-1, 122], states[-1, 129] = .01, float(goal[7]) + error
                arrays = {"states": states, "final_states": states[1260:].copy(), "goal": goal,
                          "initial_invalid": np.asarray(False), "terminal_bad_observation": np.asarray(False)}
                for key in ("actions_r", "actions_l", "ppo_actions_r", "ppo_actions_l"):
                    arrays[key] = np.zeros((1560, 51), dtype=np.float32)
                for key in ("bad_observation", "bad_action", "bad_state", "auto_reset"):
                    arrays[key] = np.zeros(1560, dtype=bool)
                if method != "baseline":
                    arrays["actions_r"][1260:, 0] = 1
                measured_error = abs(float(states[-1, 129]) - float(goal[7]))
                row = dict(copy.deepcopy(spec), method=method, valid=True,
                           target_position_m=goal[:3].tolist(), target_quaternion_wxyz=goal[3:7].tolist(),
                           target_articulation_rad=float(goal[7]), initial_articulation_rad=float(states[0, 129]),
                           lift_target_position_m=[0, 0, .15], lift_target_quaternion_wxyz=[1, 0, 0, 0],
                           require_lift_pose_success=False, timing=report.TIMING.copy(),
                           paper_articulation_error_rad=measured_error,
                           paper_position_error_m=float(states[-1, 122]), paper_orientation_error_rad=0.0,
                           paper_success=measured_error < .5, ood_goal_success=measured_error < .5,
                           strict_goal_success=measured_error < .1,
                           opening_entry={"position_error_m": 0., "orientation_error_rad": 0.,
                                          "articulation_rad": float(states[1260, 129]), "prior_lift_success": True,
                                          "prior_rollout_valid": True},
                           guidance_candidate_counts=[15, 0, 0, 0, 0, 0, 0] if method == "baseline" else [0, 15, 0, 0, 0, 0, 0])
                path = self.output / "trajectories" / method / (spec["episode_id"] + ".npz")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                row.update(trajectory_path=str(path.relative_to(self.output)), trajectory_sha256=report._sha256(path))
                self.rows.append(row)
        self.save()

    def save(self):
        (self.output / "plan.json").write_text(json.dumps(self.plan))
        manifest = {"complete": True, "plan_sha256": report._sha256(self.output / "plan.json"), "episodes": self.rows}
        (self.output / "results.json").write_text(json.dumps(manifest))

    def mutate(self, row, key, index, value):
        path = self.output / row["trajectory_path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays[key][index] = value
        np.savez_compressed(path, **arrays)
        row["trajectory_sha256"] = report._sha256(path)
        self.save()

    def test_valid_threeway_report_and_machine_table(self):
        result = report.write_report(self.output)
        self.assertTrue(result["comparison_eligible"], result["pairing_audit"]["issues"])
        self.assertTrue(result["comparison_conclusion_allowed"])
        self.assertEqual(6, result["pairing_audit"]["matching_pairs"])
        for group in ("normal", "ood_small", "ood_large"):
            self.assertEqual(1, result["groups"][group]["guided_delta"]["n"])
            self.assertEqual(1, result["groups"][group]["paired_vs_baseline"]["guided_delta"]["paper_success"]["gains"])
        self.assertEqual(2, result["groups"]["ood_articulation"]["guided_delta"]["success"]["strict_goal_success"]["count"])
        self.assertEqual(result, json.loads((self.output / "summary.json").read_text()))
        with (self.output / "metricstable.csv").open() as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual(12, len(table))
        self.assertIn("PE_m_mean", table[0])
        text = (self.output / "THREEWAY_REVIEW.md").read_text()
        self.assertIn("探索的な単一 seed", text)
        self.assertIn("AE は quaternion の SO(3)", text)
        self.assertIn("以前の PPO・absolute の結果は流用しません", text)

    def test_invalid_trial_remains_in_denominator_without_invalidating_pairing(self):
        row = self.rows[5]
        row.update(valid=False, paper_success=False, ood_goal_success=False, strict_goal_success=False)
        row.update({key: None for key in report.ERRORS})
        self.mutate(row, "auto_reset", 1400, True)
        result = report.write_report(self.output)
        self.assertTrue(result["comparison_eligible"], result["pairing_audit"]["issues"])
        group = result["groups"]["ood_articulation"]["guided_delta"]
        self.assertEqual({"count": 1, "n": 2, "rate": .5}, group["success"]["paper_success"])
        self.assertEqual(1, group["invalid"])
        self.assertEqual(1, group["errors"]["paper_position_error_m"]["excluded_or_missing"])

    def test_ppo_prefix_mismatch_withholds_all_paired_effects(self):
        self.mutate(self.rows[2], "ppo_actions_r", (1259, 0), .1)
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        self.assertIsNone(result["groups"]["normal"]["paired_vs_baseline"])
        self.assertTrue(any("ppo_actions_r" in issue for issue in result["pairing_audit"]["issues"]))
        self.assertIn("結論を保留", (self.output / "THREEWAY_REVIEW.md").read_text())

    def test_phase_reset_and_last_prefix_state_are_checked(self):
        self.mutate(self.rows[1], "final_states", (0, 13), .1)
        self.mutate(self.rows[2], "states", (1260, 122), .1)
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        issues = result["pairing_audit"]["issues"]
        self.assertTrue(any("Shared prefix differs: final_states" in issue for issue in issues))
        self.assertTrue(any("Shared prefix differs: states" in issue for issue in issues))

    def test_missing_and_duplicate_results_keep_planned_denominators(self):
        self.rows.pop(2)
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.save()
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        self.assertEqual(2, result["pairing_audit"]["unverified_slots"])
        for method in report.METHODS:
            self.assertEqual(1, result["groups"]["normal"][method]["n"])
        self.assertEqual(0, result["groups"]["normal"]["guided_delta"]["success"]["paper_success"]["count"])

    def test_corrupt_trajectory_hash_cannot_supply_success_or_errors(self):
        self.rows[2]["trajectory_sha256"] = "0" * 64
        self.save()
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        row = result["groups"]["normal"]["guided_delta"]
        self.assertEqual(1, row["unverified"])
        self.assertEqual(0, row["errors"]["paper_position_error_m"]["n"])

    def test_stored_success_is_checked_against_trajectory(self):
        self.rows[0]["paper_success"] = True
        self.save()
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        self.assertTrue(any("paper_success disagrees" in issue for issue in result["pairing_audit"]["issues"]))

    def test_predictor_hash_and_frozen_score_are_checked(self):
        self.plan["predictors"]["guided_delta"]["sha256"] = "0" * 64
        self.plan["guidance"]["horizon_steps"] = 10
        self.save()
        result = report.write_report(self.output)
        self.assertFalse(result["comparison_eligible"])
        self.assertTrue(any("Predictor SHA256" in issue for issue in result["pairing_audit"]["issues"]))
        self.assertTrue(any("fixed seven-candidate" in issue for issue in result["pairing_audit"]["issues"]))

    def test_smoke_does_not_authorize_ood_conclusions(self):
        self.plan.update(purpose="technical_smoke", split="train")
        self.save()
        result = report.write_report(self.output)
        self.assertTrue(result["comparison_eligible"], result["pairing_audit"]["issues"])
        self.assertFalse(result["comparison_conclusion_allowed"])
        self.assertIn("OOD 改善に関する結論は出しません", (self.output / "THREEWAY_REVIEW.md").read_text())


if __name__ == "__main__":
    unittest.main()

"""Controlled goal shifts, SO(3) execution, intermediate success and reporting."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from ood_suite.geometry import axis_angle_quaternion, orientation_goal_pair
from ood_suite.metrics import evaluate_episode, evaluate_goal_diagnostics, quaternion_geodesic_rad
from ood_suite.report import generate_report
from ood_suite.targets import build_plan, load_config, sample_target, select_profiles, stable_seed
from ood_suite.runtime import Controller, timing_for
import ood_record
from tools.tests.test_ood_runtime import _data, _fake_rotations, EventEnvironment


def label_set(name):
    label = dict(seq_num=1, seq_name=name + "_use_01", obj_name=name, init_frame=1, grasp_frame=2)
    return {"grasp": [dict(label, kind="two_grasp")],
            "articulation": [dict(label, kind="right_arti")]}


class MultiaxisTargetTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/archive/ood_multiaxis_legacy.json")

    def plan(self, objects=("box", "espressomachine"), samples=4, **kwargs):
        return build_plan(ROOT, self.config, objects=list(objects), samples=samples,
                          label_sets={name: label_set(name) for name in objects}, **kwargs)

    def test_suites_bound_the_work_and_include_matched_controls(self):
        self.assertEqual(len(select_profiles(self.config)), 17)
        self.assertEqual(len(select_profiles(self.config, suite="full")), 26)
        self.assertEqual(len(select_profiles(self.config, suite="all")), 29)
        for suite in self.config["suites"]:
            names = select_profiles(self.config, suite=suite)
            for name in names:
                self.assertIn(self.config["profiles"][name]["compare_to"], names)
        with self.assertRaises(ValueError):
            select_profiles(self.config, profiles=["id_control"], suite="core")
        with self.assertRaises(ValueError):
            select_profiles(self.config, suite="typo")

    def test_legacy_five_profile_targets_keep_their_original_values(self):
        config = load_config(ROOT / "configs/ood_eval.json")
        values = [sample_target(name, seed, profile) for name in config["objects"]
                  for seed in (1,2,9,1234,98765) for profile in config["profiles"].values()]
        digest = hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(digest, "f622f3b4f62204c29bdd21f76f56d82e9f66f17334e36b1bc5bd898152297f28")

    def test_effective_config_reproduces_explicit_subset(self):
        original = self.plan(profiles=["id_control", "air_pitch_large"])
        repeated = build_plan(ROOT, original["config"], label_sets={
            name: label_set(name) for name in original["objects"]})
        self.assertEqual(original["episodes"], repeated["episodes"])

    def test_direction_and_distance_do_not_change_each_other(self):
        counts = Counter()
        for i in range(100):
            seed = stable_seed(1, "box", i)
            base = sample_target("box", seed, {})
            direction = sample_target("box", seed, self.config["profiles"]["direction_balanced"], [0, math.pi], i)
            distance = sample_target("box", seed, self.config["profiles"]["distance_large"], [0, math.pi], i)
            self.assertAlmostEqual(math.hypot(*base["translation_m"]), math.hypot(*direction["translation_m"]), places=12)
            self.assertAlmostEqual(math.atan2(*reversed(base["translation_m"])),
                                   math.atan2(*reversed(distance["translation_m"])), places=12)
            self.assertTrue(0.5 <= math.hypot(*distance["translation_m"]) <= 0.8)
            counts[direction["strata"]["translation_direction_deg"]] += 1
        self.assertEqual(counts, {0: 25, 90: 25, 180: 25, 270: 25})

    def test_sign_axis_and_severity_are_paired(self):
        signs = Counter()
        for i in range(100):
            seed = stable_seed(9, "orientation", i)
            near = sample_target("box", seed, self.config["profiles"]["air_so3_medium"], [0, math.pi], i)
            far = sample_target("box", seed, self.config["profiles"]["air_so3_large"], [0, math.pi], i)
            a, b = near["orientation_rotation"], far["orientation_rotation"]
            self.assertEqual(a["axis"], b["axis"])
            self.assertGreater(abs(b["angle_rad"]), abs(a["angle_rad"]))
            self.assertAlmostEqual(sum(v*v for v in b["axis"]), 1, places=12)
            self.assertAlmostEqual(sum(v*v for v in b["quaternion_wxyz"]), 1, places=12)
            signs[far["strata"]["rotation_sign"]] += 1
        self.assertEqual(signs, {"positive": 50, "negative": 50})

    def test_air_control_and_rotations_share_all_other_targets(self):
        plan = self.plan(objects=("box",), samples=1)
        items = {item["profile"]: item for item in plan["episodes"]}
        for name in ("air_roll_large", "air_pitch_large", "air_so3_large"):
            control, target = items["air_control"], items[name]
            for field in ("grasp_label", "arti_label", "seed"):
                self.assertEqual(control[field], target[field])
            for field in ("lift_m", "translation_m", "orientation_delta_rad", "articulation_rad"):
                self.assertEqual(control["target"][field], target["target"][field])
            self.assertEqual(target["compare_to"], "air_control")

    def test_normalized_limit_is_feasible_but_not_always_training_ood(self):
        plan = self.plan(samples=20, profiles=["angle_upper", "stress_articulation_180"])
        for item in plan["episodes"]:
            if item["profile"] == "angle_upper":
                self.assertTrue(item["target_reachable"])
                self.assertEqual(item["articulation_randomized_target_ood"], item["object"] == "box")
            else:
                self.assertAlmostEqual(item["target"]["articulation_rad"], math.pi)
                self.assertEqual(item["target_reachable"], item["object"] == "box")

    def test_combinations_use_the_same_sampled_component_targets(self):
        items = {item["profile"]: item["target"] for item in self.plan(objects=("box",), samples=1)["episodes"]}
        combined = items["combo_planar"]
        self.assertEqual(combined["lift_m"], items["height_medium"]["lift_m"])
        self.assertEqual(combined["translation_m"], items["distance_medium"]["translation_m"])
        self.assertEqual(combined["orientation_rotation"], items["yaw_medium"]["orientation_rotation"])
        self.assertEqual(combined["articulation_rad"], items["angle_upper"]["articulation_rad"])
        pair = items["combo_distance_yaw"]
        self.assertEqual(pair["translation_m"], items["distance_medium"]["translation_m"])
        self.assertEqual(pair["orientation_rotation"], items["yaw_medium"]["orientation_rotation"])

    def test_invalid_rotations_and_conflicting_goals_fail_early(self):
        definitions = [
            {"orientation": {"axis": "typo"}}, {"orientation": {"angle_deg": 181}},
            {"orientation": {"stage": "typo"}}, {"orientation": {"frame": "camera"}},
            {"orientation": {"angle_deg": [-1, 2]}}, {"orientation": {}, "orientation_delta_deg": 30},
            {"articulation_deg": 90, "articulation_limit_fraction": 0.9},
            {"translation_directions_deg": []}, {"translation_direction": "typo"}, {"orientatoin": 90},
        ]
        for definition in definitions:
            with self.subTest(definition=definition), self.assertRaises(ValueError):
                sample_target("box", 1, definition, [0, math.pi], 0)


class RotationExecutionTests(unittest.TestCase):
    def test_world_and_object_rotations_are_different_and_correct(self):
        reference = axis_angle_quaternion([0, 0, 1], math.pi/2)
        delta = axis_angle_quaternion([1, 0, 0], math.pi/2)
        world, _ = orientation_goal_pair(reference, dict(quaternion_wxyz=delta, frame="world", stage="both"))
        local, _ = orientation_goal_pair(reference, dict(quaternion_wxyz=delta, frame="object", stage="both"))
        np.testing.assert_allclose(world, [0.5, 0.5, -0.5, 0.5], atol=1e-12)
        np.testing.assert_allclose(local, [0.5, 0.5, 0.5, 0.5], atol=1e-12)
        self.assertAlmostEqual(quaternion_geodesic_rad(world, local), 2*math.pi/3)

    def test_rotation_stage_preserves_untouched_goal(self):
        reference = [1, 0, 0, 0]
        delta = axis_angle_quaternion([0, 1, 0], math.pi)
        for stage in ("lift", "placement", "both"):
            placement, lift = orientation_goal_pair(reference, dict(quaternion_wxyz=delta, frame="world", stage=stage))
            self.assertAlmostEqual(quaternion_geodesic_rad(reference, placement), 0 if stage == "lift" else math.pi)
            self.assertAlmostEqual(quaternion_geodesic_rad(reference, lift), 0 if stage == "placement" else math.pi)

    def test_controller_sends_distinct_lift_and_placement_quaternions(self):
        config = load_config(ROOT / "configs/archive/ood_multiaxis_legacy.json")
        plan = build_plan(ROOT, config, objects=["box"], profiles=["air_pitch_large"], samples=1,
                          label_sets={"box": label_set("box")})
        data, env = _data(), EventEnvironment()
        with patch.dict(sys.modules, _fake_rotations()):
            controller = Controller(env, data, plan["episodes"], timing_for("box"))
        self.assertGreater(quaternion_geodesic_rad(controller.goal[0,3:7], controller.lift_goal[0,3:7]), 1.5)
        for step in (0, 560):
            env.step_number = step
            controller.phase(step)
        goals = [args[1] for step, name, args in env.events if name == "set_obj_goal"]
        np.testing.assert_array_equal(goals[0], controller.lift_goal)
        np.testing.assert_array_equal(goals[1], controller.goal)


def diagnostic_rollout():
    timing = {"protocol": "standard", "grasp_steps": 2, "transit_steps": 4, "arti_steps": 102}
    row = [0.0]*138
    row[125] = row[133] = 1
    row[3] = row[129] = row[137] = 0.55
    states = [row[:] for _ in range(117)]
    for state in states[3:7]:
        state[124] = 0.6
    spec = {"object": "box", "target": {"lift_m": 0.6}, "require_lift_pose_success": True,
            "target_position_m": [0,0,0], "target_quaternion_wxyz": [1,0,0,0],
            "target_articulation_rad": 0.55, "lift_target_position_m": [0,0,0.6],
            "lift_target_quaternion_wxyz": [0,1,0,0]}
    return states, spec, timing


class IntermediateMetricTests(unittest.TestCase):
    def test_final_paper_success_does_not_hide_failed_air_rotation(self):
        states, spec, timing = diagnostic_rollout()
        self.assertTrue(evaluate_episode(states, spec, timing)["paper_success"])
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertFalse(result["ood_goal_success"])
        self.assertFalse(result["lift_pose_success"])
        states[4][125:129] = [0,1,0,0]
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertTrue(result["ood_goal_success"])
        self.assertTrue(result["strict_goal_success"])

    def test_position_and_orientation_must_pass_at_same_time(self):
        states, spec, timing = diagnostic_rollout()
        states[4][124] = 0
        states[4][125:129] = [0,1,0,0]
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertAlmostEqual(result["lift_position_error_min_m"], 0)
        self.assertAlmostEqual(result["lift_orientation_error_min_rad"], 0)
        self.assertFalse(result["lift_pose_success"])

    def test_small_goal_exposes_initial_paper_tolerance_and_strict_failure(self):
        states, spec, timing = diagnostic_rollout()
        spec["lift_target_quaternion_wxyz"] = [1,0,0,0]
        spec["target_articulation_rad"] = 0.3
        for state in states:
            state[3] = state[129] = 0
        self.assertTrue(evaluate_episode(states, spec, timing)["paper_success"])
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertTrue(result["articulation_initial_within_paper_tolerance"])
        self.assertFalse(result["articulation_initial_within_strict_tolerance"])
        self.assertEqual(result["articulation_motion_rad"], 0)
        self.assertTrue(result["ood_goal_success"])
        self.assertFalse(result["strict_goal_success"])

    def test_articulation_start_distinguishes_motion_during_transport(self):
        states, spec, timing = diagnostic_rollout()
        states[0][129] = 0
        result = evaluate_goal_diagnostics(states, spec, timing)
        self.assertFalse(result["articulation_initial_within_paper_tolerance"])
        self.assertTrue(result["articulation_start_within_paper_tolerance"])
        self.assertAlmostEqual(result["articulation_phase_motion_rad"], 0)
        self.assertAlmostEqual(result["articulation_motion_rad"], 0.55)


class MultiaxisReportTests(unittest.TestCase):
    def make_run(self, directory, profiles, samples=1):
        config = load_config(ROOT / "configs/archive/ood_multiaxis_legacy.json")
        plan = build_plan(ROOT, config, objects=["box"], profiles=profiles, samples=samples,
                          label_sets={"box": label_set("box")})
        (directory/"plan.json").write_text(json.dumps(plan))
        rows = [dict(item, valid=True, paper_success=True, public_success=True,
                     ood_goal_success=True, strict_goal_success=True) for item in plan["episodes"]]
        return rows

    def finish(self, directory, rows):
        (directory/"episodes.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        generate_report(directory)
        return json.loads((directory/"summary.json").read_text())

    def test_air_rotation_compares_to_same_height_control(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = self.make_run(directory, ["id_control", "air_control", "air_roll_large"])
            for row in rows:
                row["paper_success"] = row["profile"] != "air_control"
            summary = self.finish(directory, rows)
            comparison = next(row for row in summary["paired_comparisons"]
                              if row["profile"] == "air_roll_large" and row["metric"] == "paper_success")
            self.assertEqual(comparison["reference_profile"], "air_control")
            self.assertEqual(comparison["success_difference_pp"], 100)
            self.assertEqual(comparison["reference_success_count"], 0)
            self.assertIsNone(comparison["target_success_given_reference_success"])

    def test_interaction_uses_four_paired_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = self.make_run(directory, ["id_control", "distance_medium", "yaw_medium", "combo_distance_yaw"])
            for row in rows:
                row["paper_success"] = row["profile"] != "combo_distance_yaw"
            summary = self.finish(directory, rows)
            comparison = next(row for row in summary["interaction_comparisons"] if row["metric"] == "paper_success")
            self.assertEqual(comparison["interaction_pp"], -100)
            self.assertEqual(comparison["complete_quadruples"], 1)

    def test_joint_ood_denominator_excludes_in_range_targets_and_keeps_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            config = load_config(ROOT / "configs/archive/ood_multiaxis_legacy.json")
            plan = build_plan(ROOT,config,objects=["box","espressomachine"],profiles=["angle_upper"],samples=2,
                              label_sets={name:label_set(name) for name in ("box","espressomachine")})
            (directory/"plan.json").write_text(json.dumps(plan))
            rows = [dict(item,valid=True,paper_success=True,public_success=True,strict_goal_success=True,
                         ood_goal_success=True) for item in plan["episodes"]]
            rows[1]["valid"] = False
            summary = self.finish(directory, rows)
            groups = {item["object"]:item for item in summary["groups"]}
            self.assertEqual(groups["box"]["joint_ood_planned"],2)
            self.assertEqual(groups["box"]["joint_ood_paper_success_rate"],0.5)
            self.assertEqual(groups["espressomachine"]["joint_ood_planned"],0)
            self.assertIsNone(groups["espressomachine"]["joint_ood_paper_success_rate"])

    def test_direction_strata_have_planned_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            rows = self.make_run(directory, ["direction_balanced"], samples=100)
            summary = self.finish(directory, rows[:-1])
            strata = summary["stratified_summaries"]
            self.assertEqual(len(strata), 4)
            self.assertEqual({row["planned"] for row in strata}, {25})
            self.assertEqual(sum(row["missing"] for row in strata), 1)
            self.assertIn("方向・回転符号の内訳", (directory/"report.md").read_text())

    def test_recording_selects_full_goal_success_and_checks_replay(self):
        rows = [dict(episode_id=str(i),object="box",profile="air_roll_large",sample_index=i,
                     valid=True,paper_success=True,public_success=True,ood_goal_success=(i==0)) for i in range(2)]
        selected = ood_record.select_episodes(rows, metric="ood_goal_success")
        self.assertEqual(selected[0]["selected"], [rows[0]])
        self.assertEqual(selected[1]["selected"], [rows[1]])
        replay = dict(rows[0], verified=True,ood_goal_success=False)
        with self.assertRaisesRegex(ValueError,"ood_goal_success"):
            ood_record.verify_replay(rows[0],replay)
        with self.assertRaisesRegex(ValueError,"no Boolean"):
            ood_record.select_episodes([dict(rows[0],ood_goal_success=None)], metric="ood_goal_success")


if __name__ == "__main__":
    unittest.main()
